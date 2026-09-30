[CmdletBinding(DefaultParameterSetName = 'Files')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Files')]
    [ValidateNotNullOrEmpty()]
    [string] $SourcePath,

    [Parameter(Mandatory = $true, ParameterSetName = 'Files')]
    [ValidateNotNullOrEmpty()]
    [string] $OutputPath,

    [Parameter(ParameterSetName = 'Files')]
    [ValidateNotNullOrEmpty()]
    [string] $Title = 'COREO YIN reverse - YAN forward',

    # Optional: keep the validated four-channel intermediate at this path.
    # If omitted, the intermediate is temporary and is removed after finalization.
    [Parameter(ParameterSetName = 'Files')]
    [string] $QuadOutputPath,

    # Optional FFmpeg executable for compressed/non-WAVE inputs or layouts
    # that need normalization to a stereo working stream.
    [Parameter(ParameterSetName = 'Files')]
    [string] $FfmpegPath,

    [Parameter(Mandatory = $true, ParameterSetName = 'StdinStdout')]
    [switch] $StdinStdout,

    [Parameter(ParameterSetName = 'StdinStdout')]
    [ValidateRange(1, 1048576)]
    [int] $FramesPerBlock = 16384,

    [Parameter(Mandatory = $true, ParameterSetName = 'StreamSelfTest')]
    [switch] $StreamSelfTest
)

# One-command post-capture pathway:
#   supported mono/stereo WAV directly, or FFmpeg-decoded source
#   -> validated four-channel YIN/YAN WAV
#   validated quad -> four-channel IEEE-float32 WAV with polarity inversion
#
# Stage 1 preserves source samples and sample encoding. YIN is written in
# reverse frame order; YAN stays forward. Mono input is duplicated into the
# left/right pair for each half. Other inputs are decoded to a stereo
# IEEE-float working WAV by FFmpeg (first audio stream; multichannel inputs
# are downmixed by FFmpeg). Stage 2 keeps all four channels in the same
# order, multiplies every sample by -1 once, and retains every frame in its
# existing order. This is signal-polarity inversion, not spatial rotation or
# binaural rendering. Stage 2 is pinned to the highest active logical
# processor number in the current Windows processor group, at Highest managed
# thread priority. Windows does not assign a special cryptography core.
#
# -StdinStdout is a separate raw-stream mode: stereo float32 little-endian in,
# four-channel float32 little-endian out. It uses a temporary scratch stream
# because the attached mapping reverses YIN over the complete finite input.

$csharpCode = @'
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

public sealed class AnythingToCoreoFloatReport
{
    public string SourcePath { get; set; }
    public string QuadPath { get; set; }
    public string OutputPath { get; set; }
    public uint SampleRate { get; set; }
    public long Frames { get; set; }
    public long OutputDataBytes { get; set; }
    public int InputChannels { get; set; }
    public int QuadChannels { get; set; }
    public int OutputChannels { get; set; }
    public string SourceEncoding { get; set; }
    public string OutputEncoding { get; set; }
    public ushort ProcessorGroup { get; set; }
    public int PinnedLogicalProcessor { get; set; }
    public string ThreadPriority { get; set; }
    public string ChannelMap { get; set; }
}

public static class AnythingToCoreoFloatPipeline
{
    private const ushort WaveFormatPcm = 1;
    private const ushort WaveFormatIeeeFloat = 3;
    private const ushort WaveFormatExtensible = 0xFFFE;
    private const ushort QuadChannels = 4;
    private const uint QuadSpeakerMask = 0x00000033; // FL | FR | BL | BR
    private const int FramesPerBlock = 16384;

    private static readonly Guid PcmSubFormat =
        new Guid("00000001-0000-0010-8000-00AA00389B71");
    private static readonly Guid FloatSubFormat =
        new Guid("00000003-0000-0010-8000-00AA00389B71");

    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessorNumber
    {
        public ushort Group;
        public byte Number;
        public byte Reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct GroupAffinity
    {
        public UIntPtr Mask;
        public ushort Group;
        public ushort Reserved0;
        public ushort Reserved1;
        public ushort Reserved2;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern void GetCurrentProcessorNumberEx(out ProcessorNumber processorNumber);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetLogicalProcessorInformationEx(
        int relationshipType, IntPtr buffer, ref uint returnedLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetThreadGroupAffinity(
        IntPtr thread,
        ref GroupAffinity groupAffinity,
        out GroupAffinity previousGroupAffinity);

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentThread();

    private sealed class WaveInfo
    {
        internal long DataOffset;
        internal uint DataLength;
        internal uint SampleRate;
        internal uint ByteRate;
        internal ushort FormatTag;
        internal ushort Channels;
        internal ushort BitsPerSample;
        internal ushort ValidBitsPerSample;
        internal ushort BlockAlign;
        internal uint ChannelMask;
        internal uint FactFrames;
        internal bool HasFact;
        internal Guid SubFormat;
    }

    private sealed class AffinityScope : IDisposable
    {
        private readonly Thread _thread;
        private readonly ThreadPriority _previousPriority;
        private GroupAffinity _previousAffinity;
        private bool _affinitySet;
        private bool _threadAffinityStarted;
        private bool _disposed;

        internal ushort ProcessorGroup { get; private set; }
        internal int LogicalProcessor { get; private set; }

        internal AffinityScope()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                throw new PlatformNotSupportedException("The final conversion affinity step requires Windows processor-group APIs.");
            if (IntPtr.Size != 8)
                throw new PlatformNotSupportedException("Run this affinity-enabled conversion in 64-bit PowerShell so Windows processor masks are represented accurately.");

            _thread = Thread.CurrentThread;
            _previousPriority = _thread.Priority;
            GroupAffinity previous = new GroupAffinity();
            try
            {
                Thread.BeginThreadAffinity();
                _threadAffinityStarted = true;
                _thread.Priority = ThreadPriority.Highest;

                ProcessorNumber current;
                GetCurrentProcessorNumberEx(out current);
                ulong activeMask = GetActiveProcessorMask(current.Group);
                if (activeMask == 0)
                    throw new InvalidOperationException("Windows reported no active logical processors in the current group.");

                // Try the highest active logical processor first (N-1 in the
                // current group's active mask), then fall back only if Windows
                // refuses that affinity for this process/thread.
                IntPtr threadHandle = GetCurrentThread();
                bool pinned = false;
                int lastError = 0;
                for (int processor = 63; processor >= 0; processor--)
                {
                    if ((activeMask & (1UL << processor)) == 0)
                        continue;
                    ulong rawMask = 1UL << processor;
                    UIntPtr mask = IntPtr.Size == 8
                        ? new UIntPtr(rawMask)
                        : new UIntPtr(checked((uint)rawMask));
                    GroupAffinity desired = new GroupAffinity
                    {
                        Mask = mask,
                        Group = current.Group
                    };
                    if (SetThreadGroupAffinity(threadHandle, ref desired, out previous))
                    {
                        ProcessorGroup = current.Group;
                        LogicalProcessor = processor;
                        _previousAffinity = previous;
                        _affinitySet = true;
                        pinned = true;
                        break;
                    }
                    lastError = Marshal.GetLastWin32Error();
                }

                if (!pinned)
                    throw new Win32Exception(lastError,
                        "Windows could not pin the final conversion worker to an active logical processor.");
            }
            catch
            {
                _thread.Priority = _previousPriority;
                if (_threadAffinityStarted)
                {
                    Thread.EndThreadAffinity();
                    _threadAffinityStarted = false;
                }
                throw;
            }
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            int restoreError = 0;
            try
            {
                if (_affinitySet && !SetThreadGroupAffinity(GetCurrentThread(), ref _previousAffinity, out _))
                    restoreError = Marshal.GetLastWin32Error();
            }
            finally
            {
                _thread.Priority = _previousPriority;
                if (_threadAffinityStarted)
                    Thread.EndThreadAffinity();
            }

            if (restoreError != 0)
                throw new Win32Exception(restoreError, "Windows could not restore the conversion thread's original processor affinity.");
        }

        private static ulong GetActiveProcessorMask(ushort requestedGroup)
        {
            // RelationGroup is 4. First query obtains the required buffer size.
            const int RelationGroup = 4;
            uint required = 0;
            bool firstCall = GetLogicalProcessorInformationEx(RelationGroup, IntPtr.Zero, ref required);
            int firstError = Marshal.GetLastWin32Error();
            if (firstCall || required == 0 || firstError != 122) // ERROR_INSUFFICIENT_BUFFER
                throw new Win32Exception(firstError,
                    "Windows could not report the active processor groups.");

            IntPtr buffer = Marshal.AllocHGlobal(checked((int)required));
            try
            {
                uint actual = required;
                if (!GetLogicalProcessorInformationEx(RelationGroup, buffer, ref actual))
                    throw new Win32Exception(Marshal.GetLastWin32Error(),
                        "Windows could not read the active processor groups.");

                long cursor = 0;
                long end = actual;
                while (cursor <= end - 8)
                {
                    IntPtr record = IntPtr.Add(buffer, checked((int)cursor));
                    int relationship = Marshal.ReadInt32(record, 0);
                    int recordSize = Marshal.ReadInt32(record, 4);
                    if (recordSize < 8 || cursor + recordSize > end)
                        throw new InvalidDataException("Windows returned a malformed processor-group record.");

                    if (relationship == RelationGroup)
                    {
                        // GROUP_RELATIONSHIP: 2 byte counts + 20 reserved bytes,
                        // then an aligned array of PROCESSOR_GROUP_INFO records.
                        int activeGroupCount = unchecked((ushort)Marshal.ReadInt16(record, 10));
                        int groupInfoOffset = 8 + 24;
                        int groupInfoStride = 40 + IntPtr.Size;
                        if (requestedGroup >= activeGroupCount
                            || groupInfoOffset + (requestedGroup + 1) * groupInfoStride > recordSize)
                            throw new InvalidOperationException("The current processor group was absent from Windows' active-group data (group="
                                + requestedGroup + ", active groups=" + activeGroupCount + ", record size=" + recordSize + ").");

                        IntPtr groupInfo = IntPtr.Add(record,
                            groupInfoOffset + requestedGroup * groupInfoStride);
                        IntPtr maskPointer = IntPtr.Add(groupInfo, 40);
                        return IntPtr.Size == 8
                            ? unchecked((ulong)Marshal.ReadInt64(maskPointer))
                            : unchecked((uint)Marshal.ReadInt32(maskPointer));
                    }

                    cursor += recordSize;
                }

                throw new InvalidOperationException("Windows did not return a processor-group relationship record.");
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

    }

    public static long RunStdinStdout(int framesPerBlock)
    {
        return TransformStereoFloat32(
            Console.OpenStandardInput(),
            Console.OpenStandardOutput(),
            framesPerBlock);
    }

    public static long TransformStereoFloat32(Stream input, Stream output, int framesPerBlock)
    {
        if (input == null)
            throw new ArgumentNullException("input");
        if (output == null)
            throw new ArgumentNullException("output");
        if (!input.CanRead)
            throw new ArgumentException("The input stream must be readable.", "input");
        if (!output.CanWrite)
            throw new ArgumentException("The output stream must be writable.", "output");
        if (framesPerBlock <= 0)
            throw new ArgumentOutOfRangeException("framesPerBlock");

        using (FileStream decoded = CreateTemporaryFloatSpool())
        {
            CopyStream(input, decoded);
            decoded.Flush();
            decoded.Position = 0;
            return TransformSeekableStereoFloat32(decoded, output, framesPerBlock);
        }
    }

    public static string[] RunStreamSelfTests()
    {
        float[] sourceSamples = new float[]
        {
            1.0f, 10.0f,
            2.0f, 20.0f,
            3.0f, 30.0f
        };
        float[] expectedSamples = new float[]
        {
            -3.0f, -30.0f, -1.0f, -10.0f,
            -2.0f, -20.0f, -2.0f, -20.0f,
            -1.0f, -10.0f, -3.0f, -30.0f
        };
        byte[] sourceBytes = new byte[sourceSamples.Length * sizeof(float)];
        Buffer.BlockCopy(sourceSamples, 0, sourceBytes, 0, sourceBytes.Length);
        var output = new MemoryStream();
        long frameCount;
        using (var input = new MemoryStream(sourceBytes, false))
            frameCount = TransformStereoFloat32(input, output, 2);

        if (frameCount != 3 || output.Length != expectedSamples.Length * sizeof(float))
            throw new InvalidOperationException("The stream transform returned an unexpected frame or byte count.");
        byte[] outputBytes = output.ToArray();
        for (int index = 0; index < expectedSamples.Length; index++)
        {
            float actual = BitConverter.ToSingle(outputBytes, index * sizeof(float));
            if (Math.Abs(actual - expectedSamples[index]) > 0.000001f)
                throw new InvalidOperationException("The stream transform changed the required COREO sample order.");
        }

        var blockSizeOne = new MemoryStream();
        using (var input = new MemoryStream(sourceBytes, false))
            TransformStereoFloat32(input, blockSizeOne, 1);
        if (!BytesEqual(outputBytes, blockSizeOne.ToArray()))
            throw new InvalidOperationException("The full-stream reverse changed with the processing block size.");

        var emptyOutput = new MemoryStream();
        using (var emptyInput = new MemoryStream())
        {
            if (TransformStereoFloat32(emptyInput, emptyOutput, 2) != 0 || emptyOutput.Length != 0)
                throw new InvalidOperationException("An empty input stream must produce an empty output stream.");
        }

        var partialOutput = new MemoryStream();
        bool partialFrameRejected = false;
        try
        {
            using (var partialInput = new MemoryStream(new byte[7], false))
                TransformStereoFloat32(partialInput, partialOutput, 2);
        }
        catch (InvalidDataException)
        {
            partialFrameRejected = true;
        }
        if (!partialFrameRejected || partialOutput.Length != 0)
            throw new InvalidOperationException("A partial stereo float32 frame must fail before writing stdout.");

        var nonFiniteOutput = new MemoryStream();
        float[] nonFiniteSamples = new float[] { Single.NaN, 0.0f };
        byte[] nonFiniteBytes = new byte[nonFiniteSamples.Length * sizeof(float)];
        Buffer.BlockCopy(nonFiniteSamples, 0, nonFiniteBytes, 0, nonFiniteBytes.Length);
        bool nonFiniteRejected = false;
        try
        {
            using (var nonFiniteInput = new MemoryStream(nonFiniteBytes, false))
                TransformStereoFloat32(nonFiniteInput, nonFiniteOutput, 2);
        }
        catch (InvalidDataException)
        {
            nonFiniteRejected = true;
        }
        if (!nonFiniteRejected || nonFiniteOutput.Length != 0)
            throw new InvalidOperationException("Non-finite samples must fail before writing stdout.");

        return new string[]
        {
            "exact whole-stream YIN reverse and YAN forward sample order",
            "polarity inversion and four-channel float32 output",
            "block-size-independent output ordering",
            "empty-stream handling and pre-output validation"
        };
    }

    private static long TransformSeekableStereoFloat32(Stream input, Stream output, int framesPerBlock)
    {
        if (input == null || !input.CanRead || !input.CanSeek)
            throw new ArgumentException("The decoded stereo float32 stream must be readable and seekable.", "input");
        if (output == null || !output.CanWrite)
            throw new ArgumentException("The output stream must be writable.", "output");
        if (framesPerBlock <= 0)
            throw new ArgumentOutOfRangeException("framesPerBlock");
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("The raw float32 stream requires little-endian byte order.");

        const int InputBlockAlign = 2 * sizeof(float);
        const int OutputChannels = 4;
        long decodedLength = input.Length;
        if (decodedLength % InputBlockAlign != 0)
            throw new InvalidDataException("FFmpeg produced a partial stereo float32 frame.");

        long totalFrames = decodedLength / InputBlockAlign;
        byte[] validationBuffer = new byte[65536 - (65536 % sizeof(float))];
        input.Position = 0;
        long remainingBytes = decodedLength;
        while (remainingBytes > 0)
        {
            int count = (int)Math.Min(validationBuffer.Length, remainingBytes);
            count -= count % sizeof(float);
            ReadExactly(input, validationBuffer, count);
            for (int offset = 0; offset < count; offset += sizeof(float))
            {
                float sample = BitConverter.ToSingle(validationBuffer, offset);
                if (!IsFinite(sample))
                    throw new InvalidDataException("The decoded input contains a non-finite float32 sample.");
            }
            remainingBytes -= count;
        }

        byte[] reverseInput = new byte[checked(framesPerBlock * InputBlockAlign)];
        byte[] forwardInput = new byte[checked(framesPerBlock * InputBlockAlign)];
        float[] outputSamples = new float[checked(framesPerBlock * OutputChannels)];
        byte[] outputBytes = new byte[checked(framesPerBlock * OutputChannels * sizeof(float))];
        long outputFrame = 0;

        while (outputFrame < totalFrames)
        {
            int blockFrames = (int)Math.Min(framesPerBlock, totalFrames - outputFrame);
            int inputBytes = checked(blockFrames * InputBlockAlign);
            long reverseStartFrame = totalFrames - outputFrame - blockFrames;

            input.Position = checked(reverseStartFrame * InputBlockAlign);
            ReadExactly(input, reverseInput, inputBytes);
            input.Position = checked(outputFrame * InputBlockAlign);
            ReadExactly(input, forwardInput, inputBytes);

            for (int frame = 0; frame < blockFrames; frame++)
            {
                int reverseOffset = (blockFrames - 1 - frame) * InputBlockAlign;
                int forwardOffset = frame * InputBlockAlign;
                int outputOffset = frame * OutputChannels;

                outputSamples[outputOffset] = -BitConverter.ToSingle(reverseInput, reverseOffset);
                outputSamples[outputOffset + 1] = -BitConverter.ToSingle(reverseInput, reverseOffset + sizeof(float));
                outputSamples[outputOffset + 2] = -BitConverter.ToSingle(forwardInput, forwardOffset);
                outputSamples[outputOffset + 3] = -BitConverter.ToSingle(forwardInput, forwardOffset + sizeof(float));
            }

            int sampleCount = checked(blockFrames * OutputChannels);
            int outputByteCount = checked(sampleCount * sizeof(float));
            Buffer.BlockCopy(outputSamples, 0, outputBytes, 0, outputByteCount);
            output.Write(outputBytes, 0, outputByteCount);
            outputFrame += blockFrames;
        }

        output.Flush();
        return totalFrames;
    }

    private static FileStream CreateTemporaryFloatSpool()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), "dungu-coreo-" + Guid.NewGuid().ToString("N") + ".f32");
        return new FileStream(
            tempPath,
            FileMode.CreateNew,
            FileAccess.ReadWrite,
            FileShare.None,
            65536,
            FileOptions.DeleteOnClose);
    }

    private static void CopyStream(Stream input, Stream output)
    {
        byte[] buffer = new byte[65536];
        int read;
        while ((read = input.Read(buffer, 0, buffer.Length)) != 0)
            output.Write(buffer, 0, read);
        output.Flush();
    }

    private static bool BytesEqual(byte[] first, byte[] second)
    {
        if (first.Length != second.Length)
            return false;
        for (int index = 0; index < first.Length; index++)
        {
            if (first[index] != second[index])
                return false;
        }
        return true;
    }

    public static AnythingToCoreoFloatReport Convert(
        string sourcePath, string outputPath, string quadOutputPath, string title,
        string provenanceSourcePath)
    {
        if (String.IsNullOrWhiteSpace(sourcePath))
            throw new ArgumentException("A source WAV path is required.", "sourcePath");
        if (String.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentException("A final output path is required.", "outputPath");
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("WAVE samples require little-endian byte order.");

        string sourceFullPath = Path.GetFullPath(sourcePath);
        string provenanceFullPath = String.IsNullOrWhiteSpace(provenanceSourcePath)
            ? sourceFullPath : Path.GetFullPath(provenanceSourcePath);
        string outputFullPath = Path.GetFullPath(outputPath);
        string requestedQuadPath = String.IsNullOrWhiteSpace(quadOutputPath)
            ? null : Path.GetFullPath(quadOutputPath);

        EnsureDistinctPath(sourceFullPath, outputFullPath, "Source and final output paths must be different.");
        if (requestedQuadPath != null)
        {
            EnsureDistinctPath(sourceFullPath, requestedQuadPath, "Source and quad paths must be different.");
            EnsureDistinctPath(outputFullPath, requestedQuadPath, "Quad and final output paths must be different.");
        }
        if (File.Exists(outputFullPath))
            throw new IOException("Refusing to overwrite an existing final WAV: " + outputFullPath);
        if (requestedQuadPath != null && File.Exists(requestedQuadPath))
            throw new IOException("Refusing to overwrite an existing quad WAV: " + requestedQuadPath);

        WaveInfo source = InspectWave(sourceFullPath);
        if (source.Channels != 1 && source.Channels != 2)
            throw new InvalidDataException("The input must be mono or stereo uncompressed PCM/IEEE-float WAVE.");
        if (source.Channels == 2 && source.FormatTag == WaveFormatExtensible
            && source.ChannelMask != 0 && source.ChannelMask != 0x00000003)
            throw new InvalidDataException("A two-channel WAVEFORMATEXTENSIBLE source must use the conventional FL/FR mask (0x3) or an unspecified mask (0).");

        long frames = source.DataLength / source.BlockAlign;
        uint quadDataLength = checked((uint)(frames * QuadChannels * (source.BitsPerSample / 8)));
        if (frames > UInt32.MaxValue)
            throw new InvalidDataException("The input exceeds the classic RIFF frame-count limit.");

        string outputDirectory = Path.GetDirectoryName(outputFullPath);
        if (String.IsNullOrEmpty(outputDirectory) || !Directory.Exists(outputDirectory))
            throw new DirectoryNotFoundException("The final output directory does not exist: " + outputDirectory);
        string quadDirectory = requestedQuadPath == null
            ? outputDirectory : Path.GetDirectoryName(requestedQuadPath);
        if (String.IsNullOrEmpty(quadDirectory) || !Directory.Exists(quadDirectory))
            throw new DirectoryNotFoundException("The quad output directory does not exist: " + quadDirectory);

        string quadTempPath = (requestedQuadPath ?? Path.Combine(quadDirectory, Path.GetFileName(outputFullPath) + ".quad"))
            + "." + Guid.NewGuid().ToString("N") + ".tmp";
        string finalTempPath = outputFullPath + "." + Guid.NewGuid().ToString("N") + ".tmp";

        try
        {
            WriteQuadIntermediate(sourceFullPath, quadTempPath, source, frames, quadDataLength,
                title, provenanceFullPath);
            WaveInfo quad = InspectWave(quadTempPath);
            ValidateQuad(quadTempPath, quad, source, frames, quadDataLength);

            string quadWorkingPath = quadTempPath;
            if (requestedQuadPath != null)
            {
                File.Move(quadTempPath, requestedQuadPath);
                quadWorkingPath = requestedQuadPath;
            }

            ushort processorGroup;
            int logicalProcessor;
            using (AffinityScope affinity = new AffinityScope())
            {
                processorGroup = affinity.ProcessorGroup;
                logicalProcessor = affinity.LogicalProcessor;
                ConvertValidQuadToFloatCoreo(quadWorkingPath, finalTempPath, source, frames,
                    title, provenanceFullPath, outputFullPath);
                ValidateFinalFloatCoreo(finalTempPath, source.SampleRate, frames);
            }

            // Publish only after the thread's original affinity and priority
            // have been restored and the complete output passed validation.
            File.Move(finalTempPath, outputFullPath);
            if (requestedQuadPath == null && File.Exists(quadTempPath))
            {
                try { File.Delete(quadTempPath); } catch { }
            }

            return new AnythingToCoreoFloatReport
            {
                SourcePath = sourceFullPath,
                QuadPath = requestedQuadPath ?? "temporary intermediate; removed after validation",
                OutputPath = outputFullPath,
                SampleRate = source.SampleRate,
                Frames = frames,
                OutputDataBytes = checked(frames * QuadChannels * sizeof(float)),
                InputChannels = source.Channels,
                QuadChannels = QuadChannels,
                OutputChannels = QuadChannels,
                SourceEncoding = source.SubFormat == PcmSubFormat ? "PCM" : "IEEE float",
                OutputEncoding = "IEEE float32",
                ProcessorGroup = processorGroup,
                PinnedLogicalProcessor = logicalProcessor,
                ThreadPriority = "Highest (managed user-mode priority)",
                ChannelMap = "quad channels 1-4 preserved in order; each sample polarity inverted once"
            };
        }
        catch
        {
            try { if (File.Exists(quadTempPath)) File.Delete(quadTempPath); } catch { }
            try { if (File.Exists(finalTempPath)) File.Delete(finalTempPath); } catch { }
            throw;
        }
    }

    public static bool CanConvertDirectly(string sourcePath)
    {
        WaveInfo source;
        try
        {
            source = InspectWave(sourcePath);
        }
        catch (InvalidDataException)
        {
            return false;
        }
        catch (EndOfStreamException)
        {
            return false;
        }

        return (source.Channels == 1 || source.Channels == 2)
            && !(source.Channels == 2 && source.FormatTag == WaveFormatExtensible
                && source.ChannelMask != 0 && source.ChannelMask != 0x00000003);
    }

    private static void EnsureDistinctPath(string first, string second, string message)
    {
        if (String.Equals(first, second, StringComparison.OrdinalIgnoreCase))
            throw new IOException(message);
    }

    private static WaveInfo InspectWave(string path)
    {
        using (FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
        using (BinaryReader reader = new BinaryReader(stream, Encoding.ASCII, true))
        {
            if (stream.Length < 12 || ReadFourCc(reader) != "RIFF")
                throw new InvalidDataException("The file is not a RIFF/WAVE file.");
            uint riffSize = reader.ReadUInt32();
            if (ReadFourCc(reader) != "WAVE")
                throw new InvalidDataException("The RIFF form type is not WAVE.");
            if (riffSize != stream.Length - 8L)
                throw new InvalidDataException("The RIFF size does not match the file length.");

            bool sawFormat = false;
            bool sawData = false;
            WaveInfo info = new WaveInfo();
            long position = 12;
            while (position <= stream.Length - 8)
            {
                stream.Position = position;
                string chunkId = ReadFourCc(reader);
                uint chunkSize = reader.ReadUInt32();
                long payloadOffset = stream.Position;
                long nextPosition = checked(payloadOffset + chunkSize + (chunkSize & 1L));
                if (nextPosition > stream.Length)
                    throw new InvalidDataException("A RIFF chunk extends past end-of-file.");

                if (chunkId == "fmt ")
                {
                    if (sawFormat || chunkSize < 16 || chunkSize > 4096)
                        throw new InvalidDataException("The fmt chunk is missing, repeated, or invalid.");
                    byte[] format = reader.ReadBytes((int)chunkSize);
                    if (format.Length != (int)chunkSize)
                        throw new EndOfStreamException("The fmt chunk is incomplete.");
                    ParseFormat(format, info);
                    sawFormat = true;
                }
                else if (chunkId == "fact")
                {
                    if (info.HasFact || chunkSize < 4)
                        throw new InvalidDataException("The fact chunk is repeated or too short.");
                    info.FactFrames = reader.ReadUInt32();
                    info.HasFact = true;
                }
                else if (chunkId == "data")
                {
                    if (sawData)
                        throw new InvalidDataException("Multiple data chunks are not supported.");
                    info.DataOffset = payloadOffset;
                    info.DataLength = chunkSize;
                    sawData = true;
                }

                position = nextPosition;
            }

            if (!sawFormat || !sawData)
                throw new InvalidDataException("The WAVE file needs one fmt chunk and one data chunk.");
            if (stream.Length - info.DataOffset != info.DataLength + (info.DataLength & 1L))
                throw new InvalidDataException("The data chunk must be final so its boundary is unambiguous.");
            if (info.SampleRate == 0 || info.Channels == 0 || info.BlockAlign == 0
                || (long)info.ByteRate != (long)info.SampleRate * info.BlockAlign)
                throw new InvalidDataException("The sample rate, byte rate, and block alignment disagree.");
            if (info.DataLength % info.BlockAlign != 0)
                throw new InvalidDataException("The data chunk does not contain complete sample frames.");

            bool isPcm = info.SubFormat == PcmSubFormat;
            bool isFloat = info.SubFormat == FloatSubFormat;
            bool pcmWidth = info.BitsPerSample == 8 || info.BitsPerSample == 16
                || info.BitsPerSample == 24 || info.BitsPerSample == 32;
            bool floatWidth = info.BitsPerSample == 32 || info.BitsPerSample == 64;
            if ((!isPcm && !isFloat) || (isPcm && !pcmWidth) || (isFloat && !floatWidth))
                throw new InvalidDataException("Supported input is uncompressed PCM 8/16/24/32-bit or IEEE float 32/64-bit WAVE.");
            if (info.ValidBitsPerSample == 0 || info.ValidBitsPerSample > info.BitsPerSample
                || (isFloat && info.ValidBitsPerSample != info.BitsPerSample)
                || info.BlockAlign != info.Channels * (info.BitsPerSample / 8))
                throw new InvalidDataException("Sample width, valid bits, and block alignment disagree.");

            long frames = info.DataLength / info.BlockAlign;
            if (isFloat && (!info.HasFact || info.FactFrames != frames))
                throw new InvalidDataException("IEEE-float WAVE requires a fact frame count matching the data.");
            if (info.HasFact && info.FactFrames != frames)
                throw new InvalidDataException("The fact frame count does not match the data.");
            return info;
        }
    }

    private static void ParseFormat(byte[] format, WaveInfo info)
    {
        info.FormatTag = BitConverter.ToUInt16(format, 0);
        info.Channels = BitConverter.ToUInt16(format, 2);
        info.SampleRate = BitConverter.ToUInt32(format, 4);
        info.ByteRate = BitConverter.ToUInt32(format, 8);
        info.BlockAlign = BitConverter.ToUInt16(format, 12);
        info.BitsPerSample = BitConverter.ToUInt16(format, 14);
        info.ValidBitsPerSample = info.BitsPerSample;

        if (info.FormatTag == WaveFormatPcm)
            info.SubFormat = PcmSubFormat;
        else if (info.FormatTag == WaveFormatIeeeFloat)
            info.SubFormat = FloatSubFormat;
        else if (info.FormatTag == WaveFormatExtensible)
        {
            if (format.Length < 40 || BitConverter.ToUInt16(format, 16) < 22)
                throw new InvalidDataException("WAVEFORMATEXTENSIBLE data is incomplete.");
            info.ValidBitsPerSample = BitConverter.ToUInt16(format, 18);
            info.ChannelMask = BitConverter.ToUInt32(format, 20);
            byte[] guidBytes = new byte[16];
            Buffer.BlockCopy(format, 24, guidBytes, 0, 16);
            info.SubFormat = new Guid(guidBytes);
        }
        else
            throw new InvalidDataException("Compressed and unknown WAVE formats are not supported.");
    }

    private static void WriteQuadIntermediate(string sourcePath, string tempPath, WaveInfo source,
        long frames, uint dataLength, string title, string provenanceSourcePath)
    {
        ushort outputBlockAlign = checked((ushort)(4 * (source.BitsPerSample / 8)));
        byte[] listPayload = CreateInfoList(title, provenanceSourcePath, source.Channels);
        long listPaddedLength = listPayload.Length + (listPayload.Length & 1L);
        long optionalFactLength = source.SubFormat == FloatSubFormat ? 12L : 0L;
        long fileLength = checked(76L + optionalFactLength + listPaddedLength + dataLength);
        long riffSize = fileLength - 8L;
        if (riffSize > UInt32.MaxValue || frames > UInt32.MaxValue)
            throw new InvalidDataException("The intermediate exceeds classic RIFF limits.");

        using (FileStream input = new FileStream(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read))
        using (FileStream output = new FileStream(tempPath, FileMode.CreateNew, FileAccess.ReadWrite,
            FileShare.Read, 65536, FileOptions.SequentialScan))
        using (BinaryWriter writer = new BinaryWriter(output, Encoding.ASCII, true))
        {
            WriteQuadHeader(writer, source, outputBlockAlign, (uint)riffSize, dataLength,
                (uint)frames, listPayload);

            byte[] reverseInput = new byte[FramesPerBlock * source.BlockAlign];
            byte[] forwardInput = new byte[FramesPerBlock * source.BlockAlign];
            byte[] quadBlock = new byte[FramesPerBlock * outputBlockAlign];
            long outputFrame = 0;

            while (outputFrame < frames)
            {
                int blockFrames = (int)Math.Min(FramesPerBlock, frames - outputFrame);
                int inputBytes = checked(blockFrames * source.BlockAlign);
                long reverseStartFrame = frames - outputFrame - blockFrames;
                input.Position = checked(source.DataOffset + reverseStartFrame * source.BlockAlign);
                ReadExactly(input, reverseInput, inputBytes);
                input.Position = checked(source.DataOffset + outputFrame * source.BlockAlign);
                ReadExactly(input, forwardInput, inputBytes);

                int bytesPerSample = source.BitsPerSample / 8;
                for (int frame = 0; frame < blockFrames; frame++)
                {
                    int reverseOffset = (blockFrames - 1 - frame) * source.BlockAlign;
                    int forwardOffset = frame * source.BlockAlign;
                    int targetOffset = frame * outputBlockAlign;

                    CopySample(reverseInput, reverseOffset, quadBlock, targetOffset, bytesPerSample);
                    CopySample(reverseInput, reverseOffset + (source.Channels == 1 ? 0 : bytesPerSample),
                        quadBlock, targetOffset + bytesPerSample, bytesPerSample);
                    CopySample(forwardInput, forwardOffset, quadBlock,
                        targetOffset + 2 * bytesPerSample, bytesPerSample);
                    CopySample(forwardInput, forwardOffset + (source.Channels == 1 ? 0 : bytesPerSample),
                        quadBlock, targetOffset + 3 * bytesPerSample, bytesPerSample);
                }

                writer.Write(quadBlock, 0, checked(blockFrames * outputBlockAlign));
                outputFrame += blockFrames;
            }

            writer.Flush();
            output.Flush(true);
        }
    }

    private static void WriteQuadHeader(BinaryWriter writer, WaveInfo source, ushort blockAlign,
        uint riffSize, uint dataLength, uint frames, byte[] listPayload)
    {
        WriteFourCc(writer, "RIFF");
        writer.Write(riffSize);
        WriteFourCc(writer, "WAVE");
        WriteFormatExtensible(writer, QuadChannels, source.SampleRate, blockAlign,
            source.BitsPerSample, source.ValidBitsPerSample, QuadSpeakerMask, source.SubFormat);

        if (source.SubFormat == FloatSubFormat)
        {
            WriteFourCc(writer, "fact");
            writer.Write((uint)4);
            writer.Write(frames);
        }

        WriteList(writer, listPayload);
        WriteFourCc(writer, "data");
        writer.Write(dataLength);
    }

    private static void ConvertValidQuadToFloatCoreo(string quadPath, string finalTempPath,
        WaveInfo source, long frames, string title, string provenanceSourcePath, string finalPath)
    {
        WaveInfo quad = InspectWave(quadPath);
        uint dataLength = checked((uint)(frames * QuadChannels * sizeof(float)));
        byte[] listPayload = CreateFinalInfoList(title, quadPath, finalPath, provenanceSourcePath);
        long listPaddedLength = listPayload.Length + (listPayload.Length & 1L);
        long fileLength = checked(88L + listPaddedLength + dataLength);
        long riffSize = fileLength - 8L;
        if (riffSize > UInt32.MaxValue || frames > UInt32.MaxValue)
            throw new InvalidDataException("The final float WAV exceeds classic RIFF limits.");

        using (FileStream input = new FileStream(quadPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        using (FileStream output = new FileStream(finalTempPath, FileMode.CreateNew, FileAccess.ReadWrite,
            FileShare.Read, 65536, FileOptions.SequentialScan))
        using (BinaryWriter writer = new BinaryWriter(output, Encoding.ASCII, true))
        {
            WriteFinalHeader(writer, source.SampleRate, (uint)riffSize, dataLength,
                (uint)frames, listPayload);
            input.Position = quad.DataOffset;

            byte[] inputBlock = new byte[FramesPerBlock * quad.BlockAlign];
            float[] outputSamples = new float[FramesPerBlock * QuadChannels];
            byte[] outputBytes = new byte[FramesPerBlock * QuadChannels * sizeof(float)];
            long frameOffset = 0;

            while (frameOffset < frames)
            {
                int blockFrames = (int)Math.Min(FramesPerBlock, frames - frameOffset);
                int inputBytes = checked(blockFrames * quad.BlockAlign);
                ReadExactly(input, inputBlock, inputBytes);

                for (int frame = 0; frame < blockFrames; frame++)
                {
                    int sourceOffset = frame * quad.BlockAlign;
                    int bytesPerSample = quad.BitsPerSample / 8;
                    for (int channel = 0; channel < QuadChannels; channel++)
                    {
                        float value = ReadSampleAsFloat(inputBlock,
                            sourceOffset + channel * bytesPerSample, quad);
                        float inverted = value * -1.0f;
                        if (!IsFinite(inverted))
                            throw new InvalidDataException("A quad sample became non-finite during float conversion.");
                        outputSamples[frame * QuadChannels + channel] = inverted;
                    }
                }

                int sampleCount = checked(blockFrames * QuadChannels);
                int byteCount = checked(sampleCount * sizeof(float));
                Buffer.BlockCopy(outputSamples, 0, outputBytes, 0, byteCount);
                writer.Write(outputBytes, 0, byteCount);
                frameOffset += blockFrames;
            }

            writer.Flush();
            output.Flush(true);
        }
    }

    private static void ValidateQuad(string path, WaveInfo quad, WaveInfo source, long frames, uint dataLength)
    {
        if (quad.FormatTag != WaveFormatExtensible || quad.Channels != QuadChannels
            || quad.ChannelMask != QuadSpeakerMask || quad.SampleRate != source.SampleRate
            || quad.SubFormat != source.SubFormat || quad.BitsPerSample != source.BitsPerSample
            || quad.ValidBitsPerSample != source.ValidBitsPerSample
            || quad.BlockAlign != 4 * (source.BitsPerSample / 8)
            || quad.DataLength != dataLength || quad.DataLength / quad.BlockAlign != frames)
            throw new InvalidDataException("The generated intermediate did not pass the expected quad layout validation.");
        if (source.SubFormat == FloatSubFormat && (!quad.HasFact || quad.FactFrames != frames))
            throw new InvalidDataException("The quad IEEE-float fact chunk is missing or has the wrong frame count.");

        // A structurally valid float WAVE can still contain NaN, infinity, or
        // float64 values that cannot be represented in the final float32 file.
        using (FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            stream.Position = quad.DataOffset;
            byte[] block = new byte[65536 - (65536 % quad.BlockAlign)];
            uint remaining = quad.DataLength;
            while (remaining > 0)
            {
                int count = (int)Math.Min((uint)block.Length, remaining);
                count -= count % quad.BlockAlign;
                ReadExactly(stream, block, count);
                for (int offset = 0; offset < count; offset += quad.BitsPerSample / 8)
                    ReadSampleAsFloat(block, offset, quad);
                remaining -= (uint)count;
            }
        }
    }

    private static float ReadSampleAsFloat(byte[] buffer, int offset, WaveInfo format)
    {
        if (format.SubFormat == FloatSubFormat)
        {
            if (format.BitsPerSample == 32)
            {
                float value = BitConverter.ToSingle(buffer, offset);
                if (!IsFinite(value))
                    throw new InvalidDataException("The quad contains NaN or infinity; it is not clean for COREO output.");
                return value;
            }

            double value64 = BitConverter.ToDouble(buffer, offset);
            if (Double.IsNaN(value64) || Double.IsInfinity(value64)
                || value64 > Single.MaxValue || value64 < -Single.MaxValue)
                throw new InvalidDataException("The quad contains a non-finite or out-of-range float64 sample.");
            return (float)value64;
        }

        int raw;
        if (format.BitsPerSample == 8)
            raw = buffer[offset] - 128;
        else if (format.BitsPerSample == 16)
            raw = BitConverter.ToInt16(buffer, offset);
        else if (format.BitsPerSample == 24)
        {
            raw = buffer[offset] | (buffer[offset + 1] << 8) | (buffer[offset + 2] << 16);
            if ((raw & 0x00800000) != 0)
                raw |= unchecked((int)0xFF000000);
        }
        else
            raw = BitConverter.ToInt32(buffer, offset);

        int paddingBits = format.BitsPerSample - format.ValidBitsPerSample;
        if (paddingBits > 0)
            raw >>= paddingBits;
        return (float)(raw / Math.Pow(2.0, format.ValidBitsPerSample - 1));
    }

    private static void ValidateFinalFloatCoreo(string path, uint sampleRate, long expectedFrames)
    {
        WaveInfo output = InspectWave(path);
        uint expectedDataLength = checked((uint)(expectedFrames * QuadChannels * sizeof(float)));
        if (output.FormatTag != WaveFormatExtensible || output.Channels != QuadChannels
            || output.SampleRate != sampleRate || output.BitsPerSample != 32
            || output.ValidBitsPerSample != 32 || output.SubFormat != FloatSubFormat
            || output.ChannelMask != QuadSpeakerMask || output.BlockAlign != 16
            || output.DataLength != expectedDataLength || output.DataLength / output.BlockAlign != expectedFrames
            || !output.HasFact || output.FactFrames != expectedFrames)
            throw new InvalidDataException("The final WAV did not pass the four-channel IEEE-float32 COREO header checks.");

        using (FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
        using (BinaryReader reader = new BinaryReader(stream, Encoding.ASCII, true))
        {
            stream.Position = output.DataOffset;
            byte[] block = new byte[65536 - (65536 % output.BlockAlign)];
            uint remaining = output.DataLength;
            while (remaining > 0)
            {
                int count = (int)Math.Min((uint)block.Length, remaining);
                count -= count % output.BlockAlign;
                int read = ReadExactly(reader, block, count);
                for (int offset = 0; offset < read; offset += sizeof(float))
                {
                    float value = BitConverter.ToSingle(block, offset);
                    if (!IsFinite(value))
                        throw new InvalidDataException("The final WAV contains a non-finite IEEE-float sample.");
                }
                remaining -= (uint)read;
            }
        }
    }

    private static void WriteFinalHeader(BinaryWriter writer, uint sampleRate, uint riffSize,
        uint dataLength, uint frames, byte[] listPayload)
    {
        WriteFourCc(writer, "RIFF");
        writer.Write(riffSize);
        WriteFourCc(writer, "WAVE");
        WriteFormatExtensible(writer, QuadChannels, sampleRate, 16, 32, 32,
            QuadSpeakerMask, FloatSubFormat);
        WriteFourCc(writer, "fact");
        writer.Write((uint)4);
        writer.Write(frames);
        WriteList(writer, listPayload);
        WriteFourCc(writer, "data");
        writer.Write(dataLength);
    }

    private static void WriteFormatExtensible(BinaryWriter writer, ushort channels, uint sampleRate,
        ushort blockAlign, ushort bitsPerSample, ushort validBitsPerSample, uint channelMask, Guid subFormat)
    {
        WriteFourCc(writer, "fmt ");
        writer.Write((uint)40);
        writer.Write(WaveFormatExtensible);
        writer.Write(channels);
        writer.Write(sampleRate);
        writer.Write(checked(sampleRate * blockAlign));
        writer.Write(blockAlign);
        writer.Write(bitsPerSample);
        writer.Write((ushort)22);
        writer.Write(validBitsPerSample);
        writer.Write(channelMask);
        writer.Write(subFormat.ToByteArray());
    }

    private static byte[] CreateInfoList(string title, string sourcePath, ushort sourceChannels)
    {
        string sourceMap = sourceChannels == 1
            ? "mono working input duplicated into left/right pairs"
            : "stereo working channels kept in order";
        using (MemoryStream memory = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(memory, Encoding.ASCII, true))
        {
            WriteFourCc(writer, "INFO");
            WriteInfoString(writer, "INAM", CleanInfoText(title) + " - validated quad intermediate");
            WriteInfoString(writer, "ISFT", "DUNGU Anything-to-COREO pipeline");
            WriteInfoString(writer, "ICMT",
                "Stage 1 channel order: 1=YIN reverse L, 2=YIN reverse R, "
                + "3=YAN forward L, 4=YAN forward R; working input was " + sourceMap + ". Source: "
                + Path.GetFileName(sourcePath) + ".");
            writer.Flush();
            return memory.ToArray();
        }
    }

    private static byte[] CreateFinalInfoList(string title, string quadPath, string finalPath,
        string provenanceSourcePath)
    {
        using (MemoryStream memory = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(memory, Encoding.ASCII, true))
        {
            WriteFourCc(writer, "INFO");
            WriteInfoString(writer, "INAM", CleanInfoText(title) + " - 4-channel inverted float32");
            WriteInfoString(writer, "ISFT", "DUNGU Anything-to-COREO pipeline");
            WriteInfoString(writer, "ICMT",
                "Four-channel IEEE float32 COREO. Channels 1-4 retain valid quad channel order. "
                + "Each sample is multiplied by -1 exactly once; final frame order is unchanged. "
                + "This is polarity inversion, not spatial rotation. Intermediate: "
                + Path.GetFileName(quadPath) + "; source: " + Path.GetFileName(provenanceSourcePath)
                + "; output: " + Path.GetFileName(finalPath) + ".");
            writer.Flush();
            return memory.ToArray();
        }
    }

    private static string CleanInfoText(string value)
    {
        return (value ?? String.Empty).Replace("\0", String.Empty);
    }

    private static void WriteInfoString(BinaryWriter writer, string id, string value)
    {
        byte[] bytes = Encoding.ASCII.GetBytes(value + "\0");
        WriteFourCc(writer, id);
        writer.Write((uint)bytes.Length);
        writer.Write(bytes);
        if ((bytes.Length & 1) != 0)
            writer.Write((byte)0);
    }

    private static void WriteList(BinaryWriter writer, byte[] listPayload)
    {
        WriteFourCc(writer, "LIST");
        writer.Write((uint)listPayload.Length);
        writer.Write(listPayload);
        if ((listPayload.Length & 1) != 0)
            writer.Write((byte)0);
    }

    private static void CopySample(byte[] source, int sourceOffset, byte[] destination,
        int destinationOffset, int bytesPerSample)
    {
        Buffer.BlockCopy(source, sourceOffset, destination, destinationOffset, bytesPerSample);
    }

    private static void ReadExactly(Stream stream, byte[] buffer, int count)
    {
        int offset = 0;
        while (offset < count)
        {
            int read = stream.Read(buffer, offset, count - offset);
            if (read == 0)
                throw new EndOfStreamException("The audio data ended before its declared length.");
            offset += read;
        }
    }

    private static int ReadExactly(BinaryReader reader, byte[] buffer, int count)
    {
        int offset = 0;
        while (offset < count)
        {
            int read = reader.Read(buffer, offset, count - offset);
            if (read == 0)
                throw new EndOfStreamException("The final float data ended before its declared length.");
            offset += read;
        }
        return offset;
    }

    private static string ReadFourCc(BinaryReader reader)
    {
        byte[] bytes = reader.ReadBytes(4);
        if (bytes.Length != 4)
            throw new EndOfStreamException("A RIFF chunk identifier is incomplete.");
        return Encoding.ASCII.GetString(bytes);
    }

    private static void WriteFourCc(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.ASCII.GetBytes(value);
        if (bytes.Length != 4)
            throw new ArgumentException("A RIFF chunk identifier must contain four ASCII bytes.", "value");
        writer.Write(bytes);
    }

    private static bool IsFinite(float value)
    {
        return !Single.IsNaN(value) && !Single.IsInfinity(value);
    }
}
'@

if ($null -eq ('AnythingToCoreoFloatPipeline' -as [type])) {
    Add-Type -TypeDefinition $csharpCode -ErrorAction Stop
}

if ($PSCmdlet.ParameterSetName -eq 'StreamSelfTest') {
    foreach ($check in [AnythingToCoreoFloatPipeline]::RunStreamSelfTests()) {
        Write-Output "PASS $check"
    }
    Write-Output 'All stdin/stdout transform tests passed.'
    return
}

if ($PSCmdlet.ParameterSetName -eq 'StdinStdout') {
    try {
        $frameCount = [AnythingToCoreoFloatPipeline]::RunStdinStdout($FramesPerBlock)
        [Console]::Error.WriteLine(
            "COREO stream complete: $frameCount stereo float32 frames transformed to raw 4-channel float32 stdout."
        )
        return
    }
    catch {
        $errorToReport = $_.Exception.GetBaseException()
        [Console]::Error.WriteLine("error: $($errorToReport.Message)")
        exit 1
    }
}

$originalSourcePath = [IO.Path]::GetFullPath($SourcePath)
$workingSourcePath = $originalSourcePath
$decodedWorkingPath = $null
$decodedByFfmpeg = $false

try {
    if (-not [AnythingToCoreoFloatPipeline]::CanConvertDirectly($originalSourcePath)) {
        $ffmpegExecutable = $null
        if (-not [string]::IsNullOrWhiteSpace($FfmpegPath)) {
            if (Test-Path -LiteralPath $FfmpegPath -PathType Leaf) {
                $ffmpegExecutable = [IO.Path]::GetFullPath($FfmpegPath)
            }
            else {
                $ffmpegCommand = Get-Command -Name $FfmpegPath -ErrorAction SilentlyContinue |
                    Select-Object -First 1
                if ($null -ne $ffmpegCommand -and $ffmpegCommand.CommandType -eq 'Application') {
                    $ffmpegExecutable = $ffmpegCommand.Source
                }
            }
        }
        else {
            $ffmpegCommand = Get-Command -Name 'ffmpeg' -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($null -ne $ffmpegCommand -and $ffmpegCommand.CommandType -eq 'Application') {
                $ffmpegExecutable = $ffmpegCommand.Source
            }
        }

        if ([string]::IsNullOrWhiteSpace($ffmpegExecutable)) {
            throw 'This source is outside the direct WAV formats. Install FFmpeg or pass -FfmpegPath to decode it; direct input supports mono/stereo PCM or IEEE-float WAV.'
        }

        $fullOutputPath = [IO.Path]::GetFullPath($OutputPath)
        $outputDirectory = [IO.Path]::GetDirectoryName($fullOutputPath)
        if (-not [IO.Directory]::Exists($outputDirectory)) {
            throw "The final output directory does not exist: $outputDirectory"
        }

        $decodedWorkingPath = Join-Path $outputDirectory (
            [IO.Path]::GetFileNameWithoutExtension($fullOutputPath) + '.decode-' +
            [guid]::NewGuid().ToString('N') + '.wav'
        )
        $startInfo = [Diagnostics.ProcessStartInfo]::new()
        $startInfo.FileName = $ffmpegExecutable
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardError = $true
        $arguments = @(
            '-hide_banner', '-nostdin', '-v', 'error', '-n',
            '-i', $originalSourcePath,
            '-map', '0:a:0', '-vn', '-sn', '-dn', '-ac', '2',
            '-c:a', 'pcm_f32le', '-f', 'wav', $decodedWorkingPath
        )
        foreach ($argument in $arguments) {
            $startInfo.ArgumentList.Add([string]$argument)
        }

        $process = [Diagnostics.Process]::Start($startInfo)
        try {
            $ffmpegError = $process.StandardError.ReadToEnd()
            $process.WaitForExit()
            if ($process.ExitCode -ne 0) {
                $detail = $ffmpegError.Trim()
                if ($detail.Length -gt 3000) { $detail = $detail.Substring($detail.Length - 3000) }
                throw "FFmpeg could not decode the first audio stream (exit $($process.ExitCode)): $detail"
            }
        }
        finally {
            $process.Dispose()
        }

        if (-not (Test-Path -LiteralPath $decodedWorkingPath -PathType Leaf)) {
            throw 'FFmpeg reported success but did not create the decoded working WAV.'
        }
        $workingSourcePath = $decodedWorkingPath
        $decodedByFfmpeg = $true
    }

    $report = [AnythingToCoreoFloatPipeline]::Convert(
        $workingSourcePath, $OutputPath, $QuadOutputPath, $Title, $originalSourcePath
    )
    $report.SourcePath = $originalSourcePath
    if ($decodedByFfmpeg) {
        $report.SourceEncoding = 'FFmpeg-decoded first audio stream to stereo IEEE float32'
    }
    $report | Format-List
}
finally {
    if ($null -ne $decodedWorkingPath -and (Test-Path -LiteralPath $decodedWorkingPath -PathType Leaf)) {
        Remove-Item -LiteralPath $decodedWorkingPath -Force
    }
}
