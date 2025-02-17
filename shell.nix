let
  pkgs = import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  };

  pythonEnv = pkgs.python311.withPackages (ps: with ps; [
    pandas
    numpy
    scikit-learn
    pytorch-bin
    torchvision-bin
    torchaudio-bin
    pip
    yfinance
    matplotlib
    requests
    pytest
    pytest-cov
    pytest-xdist
    setuptools
    filelock
    sympy
    networkx
    jinja2
    typing-extensions
  ]);

in pkgs.mkShell {
  buildInputs = with pkgs; [
    pythonEnv
#    cudaPackages.cudatoolkit
#    cudaPackages.cudnn
#    autoAddDriverRunpath
  ];

  shellHook = ''
    export PYTHONPATH="$PWD:${pythonEnv}/${pythonEnv.sitePackages}:$PYTHONPATH"
    export CUDA_VISIBLE_DEVICES=""  # Use CPU for testing
  '';
}
