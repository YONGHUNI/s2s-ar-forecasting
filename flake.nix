{
  description = "S2S atmospheric river forecasting with Nix-provided Pixi";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  # Keep Pixi independently pinned; monthly template updates do not change this project.
  # This revision was confirmed cache-installable on Sapelo2.
  inputs.nixpkgs-pixi.url = "github:NixOS/nixpkgs/4fe25e12eb9f4c744966dd3de5a05246cf098601";

  outputs = { nixpkgs, nixpkgs-pixi, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      pixiPackage = nixpkgs-pixi.legacyPackages.${system}.pixi;
    in
    {
      packages.${system}.pixi = pixiPackage;

      apps.${system}.pixi = {
        type = "app";
        program = "${pixiPackage}/bin/pixi";
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.bashInteractive
          pixiPackage
        ];

        shellHook = ''
          # Nixpkgs 26.05 wraps Pixi 0.68.1 with TLS=system, which that
          # Pixi release does not recognize. Preserve explicit webpki/all.
          if ${if pkgs.lib.versionOlder pixiPackage.version "0.69.0" then "true" else "false"}; then
            case "''${PIXI_TLS_ROOT_CERTS:-}" in
              ""|system) export PIXI_TLS_ROOT_CERTS=native ;;
            esac
          elif [ "''${PIXI_TLS_ROOT_CERTS:-}" = native ]; then
            export PIXI_TLS_ROOT_CERTS=system
          fi

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
