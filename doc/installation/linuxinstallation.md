### Linux installation proposal

<!-- Version: 2018-07-19 DWW -->

In the text below **USER** depicts the actual user name and **X** the disk partition for the workspace

##### System

- Install a long-term supported (LTS) Linux
- Consider a separate disk partition with directory **/X** besides the default /home/**USER** directory, 
  change permission after completion of Linux installation
 
      sudo chown USER /X
      ln -s /X /home/USER/X

- Optional: password for root (for login as root)

      sudo passwd root 

- Update Linux

      sudo apt update && sudo apt -y upgrade

- Optional: Folder encryption

      sudo add-apt-repository ppa:gencfsm && sudo apt update && sudo apt -y install ecryptfs-utils gnome-encfs-manager

- Optional: Monitor hardware

      sudo apt install cpufreq-utils
      sudo apt install conky

- Basic tools

      sudo apt -y install cmake unzip pkg-config git 
    
##### Python 3

    python3 -V
    sudo apt -y install python3-pip python-pip build-essential libssl-dev libffi-dev python3-dev python3-tk
    sudo apt -y install libopenblas-dev liblapack-dev libhdf5-serial-dev python-h5py python-yaml

    mpicc -v   
        # if mpicc not installed: 
        #     sudo apt -y install lam4-dev libmpich-dev libopenmpi-dev
    
    pip3 install --user mpi4py
        # if pip3 fails: 
        #     sudo python3 -m pip uninstall pip && sudo apt install python3-pip --reinstall
        #     pip3 install --user mpi4py
    
    pip3 install --user futures traitsui openpyxl xlrd numba
    
##### Spyder (python development)

    pip3 install --user rope && sudo apt -y install spyder3

##### Neural networks

    pip3 install --user neurolab
    pip3 install --user tensorflow tensorflow-gpu
    pip3 install --user keras pydot-ng
    sudo apt -y install graphviz

##### Fenics (finite elements)

    sudo add-apt-repository ppa:fenics-packages/fenics && sudo apt update && sudo apt -y install fenics

##### FiPy (finite volumes)

    pip2 install --upgrade pip setuptools
    pip2 install --user fipy

##### Mesh generation

    sudo apt -y install gmsh 
    pip3 install --user netgen
    
&nbsp; See: [Correction for netgen on Ubuntu 16.04 LTS](https://sourceforge.net/p/netgen-mesher/discussion/905307/thread/946ccfc2/), (in file _/usr/share/netgen/drawing.tcl_ change _-indirect true_ to _-indirect false_)
    
##### Genetic algorithms

    pip3 install --user modestga

##### Multigrid solver

    pip3 install --user pyamg

##### Graphics

    sudo apt -y install vtk6 python-vtk6 && pip3 install --user mayavi
    sudo apt -y install mayavi2
    
    
##### GUI

    sudo apt -y install python-wxtools python-traitsui               # loop in wx
    
    # Qt-designer as alternative to TraitsUi: 
    sudo apt -y install python-qt4 qt4-designer        # only if GUI design as *.ui file

##### Wiki

    sudo apt -y install retext

##### Text processing

    sudo apt -y install texlive texlive-science texmaker

##### Administration

    sudo apt -y install fslint                          # find duplicate files
    
##### Docker

    sudo apt remove docker docker-engine docker.io
    sudo apt install apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt install docker-ce

    sudo docker run hello-world
    
   [Instructions](https://www.howtoforge.com/tutorial/how-to-create-docker-images-with-dockerfile/#step-installing-docker)
