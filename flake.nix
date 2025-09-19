{
  inputs = {
    # nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    # systems.url = "github:nix-systems/default";
  };

  outputs =
    { nixpkgs, ... }:
    let
      eachSystem =
        f:
        nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = eachSystem (pkgs: {
        default = pkgs.mkShell {
          buildInputs = [
            pkgs.python3
            pkgs.uv
            pkgs.basedpyright
          ];
          LD_LIBRARY_PATH = "/run/opengl-driver/lib:/nix/store/zwah01qc92k4f7nywhg6chfdliv22px1-onnxruntime-1.18.1/lib:/run/current-system/sw/share/nix-ld/lib";
        };
      });
    };
}