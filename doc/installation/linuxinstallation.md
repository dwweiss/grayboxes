### Linux installation proposal

<!-- Version: 2018-06-19 DWW -->

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

      sudo apt-get update && sudo apt-get -y upgrade

- Optional: Folder encryption

      sudo add-apt-repository ppa:gencfsm && sudo apg-get update  && sudo apt-get install -y ecryptfs-utils gnome-encfs-manager

- Optional: Monitor hardware

      sudo apt-get install cpufreq-info
      sudo apt-get install conky

##### Python 3

    python3 -V
    sudo apt-get -y install -y python3-pip python-pip build-essential libssl-dev libffi-dev python3-dev python3-tk
    pip3 install --upgrade --user pip 
    pip3 install --user mpi4py futures traitsui

##### Spyder (python development)

    sudo apt-get install spyder3
    pip3 install --user rope

##### Neural networks

    pip3 install --user neurolab
    pip3 install --user tensorflow
    pip3 install --user keras

##### Fenics (finite elements)

    sudo add-apt-repository ppa:fenics-packages/fenics && sudo apt-get update && sudo apt-get install fenics

##### FiPy (finite volumes)

    sudo apt-get install python-pip
    pip2 install --upgrade pip setuptools
    pip2 install --user fipy

##### Mesh generation

    sudo apt-get install gmsh netgen
    
&nbsp; See: [Correction for netgen on Ubuntu 16.04 LTS](https://sourceforge.net/p/netgen-mesher/discussion/905307/thread/946ccfc2/), (in file _/usr/share/netgen/drawing.tcl_ change _-indirect true_ to _-indirect false_)
    
##### Genetic algorithms

    pip3 install --user modestga

##### Multigrid solver

    pip3 install --user pyamg

##### Graphics

    pip3 install --user mayavi

##### Wiki

    sudo apt-get install retext

##### Text processing

    sudo apt-get install texlive texlive-science texmaker

