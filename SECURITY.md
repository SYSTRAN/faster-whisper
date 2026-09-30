# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities through this repository's private
GitHub security advisory channel when it is enabled. Do not include secrets,
private audio, or personal data in a public issue. If private reporting is not
available, contact the maintainers privately through GitHub and provide only
the minimum information needed to reproduce the issue.

## Secure development and use

- Keep Python and runtime dependencies supported and review dependency updates
  before deployment.
- Validate input format, channel count, sample rate, and duration at trust
  boundaries. Apply resource limits when decoding untrusted audio.
- Do not log, publish, or commit recordings, transcripts, credentials, or
  personally identifying metadata without an appropriate basis and permission.
- Add regression tests for security-relevant parsing and boundary conditions;
  review generated validation artifacts before sharing them.
- Use code review and CI checks for changes, and record the requirements,
  tests, and evidence that support each behavior change.

These practices are informed by secure software lifecycle principles and
relevant CISA guidance. They are not a claim of CISA certification or GAMP 4
validation. GAMP 4 is guidance for regulated computerized systems; this
repository has not been qualified or validated for a regulated use.

## Warranty and liability

This security policy does not create a release of liability or modify the
project's license. The terms in `LICENSE` govern use of the software.
