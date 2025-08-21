{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      with pkgs.python3.pkgs;
      let
        jaxlib' = if jaxlib.meta.broken then jaxlib-bin else jaxlib;
      in
      {
        defaultPackage = buildPythonPackage {
          pname = "torch2jax";
          # Don't forget to also update the version in pyproject.toml!
          version = "0.1.0";
          pyproject = true;
          src = ./.;
          dependencies = [
            jax
            torch
          ];
          nativeCheckInputs = [
            chex
            jaxlib'
            pytestCheckHook
            torchvision
            pkgs.writableTmpDirAsHomeHook # torchvision downloads models into HOME.
          ];

          pythonImportsCheck = [ "torch2jax" ];
        };

        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.act
            pkgs.ruff

            build
            chex
            ipython
            jax
            jaxlib'
            pytest
            torch
            torchvision
            twine
          ];
        };
      }
    );
}
