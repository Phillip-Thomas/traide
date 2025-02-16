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
  ]);

in pkgs.mkShell {
  buildInputs = with pkgs; [
    pythonEnv
#    cudaPackages.cudatoolkit
#    cudaPackages.cudnn
#    autoAddDriverRunpath
  ];

shellHook = ''
  export PYTHONPATH="${pythonEnv}/${pythonEnv.sitePackages}:$PYTHONPATH"
'';
}
