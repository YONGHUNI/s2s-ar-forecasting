{
  description = "Python data science environment with Nix-provided Pixi";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      packages.${system}.pixi = pkgs.pixi;

      apps.${system}.pixi = {
        type = "app";
        program = "${pkgs.pixi}/bin/pixi";
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.bashInteractive
          pkgs.pixi
        ];

        shellHook = ''
          export PIXI_CACHE_DIR="/lscratch/$USER/pixi-cache"

          mkdir -p \
            "$PIXI_CACHE_DIR" \
            "/lscratch/$USER/pixi-envs" \
            ".pixi"

          printf 'detached-environments = "/lscratch/%s/pixi-envs"\n' "$USER" \
            > .pixi/config.toml

          # Earth2Studio
          export EARTH2STUDIO_CACHE="/lscratch/$USER/earth2studio"
          export EARTH2STUDIO_MODEL_CACHE="/work/whlab/$USER/cache/earth2studio/models"
          export EARTH2STUDIO_DATA_CACHE="/lscratch/$USER/earth2studio/data"
          export JAX_COMPILATION_CACHE_DIR="/lscratch/$USER/jax-cache"

          mkdir -p \
            "$EARTH2STUDIO_CACHE" \
            "$EARTH2STUDIO_MODEL_CACHE" \
            "$EARTH2STUDIO_DATA_CACHE" \
            "$JAX_COMPILATION_CACHE_DIR"

          if [ -f /etc/NIXOS ] && [ ! -e /lib64/ld-linux-x86-64.so.2 ]; then
            echo "python-pixi: NixOS needs programs.nix-ld.enable = true for Conda binaries." >&2
          fi
        '';

      };
    };
}
