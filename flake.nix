{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
      in with pkgs.python3.pkgs;
      let jaxlib' = if jaxlib.meta.broken then jaxlib-bin else jaxlib;
      in {
        defaultPackage = buildPythonPackage {
          pname = "torch2jax";
          version = "0.0.1";
          pyproject = true;
          src = ./.;
          propagatedBuildInputs = [ jax torch ];
          nativeCheckInputs = [ jaxlib' pytestCheckHook torchvision ];

          # torchvision downloads models into HOME.
          preCheck = ''
            export HOME=$(mktemp -d)
          '';

          pythonImportsCheck = [ "torch2jax" ];
        };

        devShell = pkgs.mkShell {
          buildInputs = [
            build
            ipython
            isort
            jax
            jaxlib'
            pytest
            torch
            torchvision
            twine
          ];
        };
      });
}
