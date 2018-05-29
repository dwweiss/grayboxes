### Linux installation proposal

<!-- Version: 2018-03-20 Dietmar Wilhelm Weiss -->

In the text below **USER** depicts the actual user name and **X** the disk partition for models etc.

##### System

- Install a long-term supported (LTS) Linux
- Consider a separate disk partition with directory **/X** besides the default /home/**USER** directory, 
  change permission after completion of Linux installation
  
      sudo chown USER /X
      ln -s /X /home/USER/X

     
- Optionally: password for root (for login as root)

      sudo passwd root 

- Update Linux

      sudo apt-get update
      sudo apt-get -y upgrade

##### Python 3

    python3 -V
    sudo apt-get install -y python3-pip
    sudo -H pip3 install --upgrade pip 
    sudo apt-get -y install build-essential libssl-dev libffi-dev python3-dev
 
    sudo apt-get install python3-tk
    sudo -H pip3 install mpi4py
    sudo -H pip3 install futures
    sudo -H pip3 install traitsui

##### Spyder (python development)

    sudo apt-get install spyder3

##### Neural networks

    sudo -H pip3 install neurolab
    sudo -H pip3 install tensorflow
    sudo -H pip3 install keras

##### Fenics (finite elements)

    sudo add-apt-repository ppa:fenics-packages/fenics
    sudo apt-get update
    sudo apt-get install fenics

##### Mesh generation

    sudo apt-get install gmsh netgen
    
&nbsp; See: [Correction for netgen on Ubuntu 16.04 LTS](https://sourceforge.net/p/netgen-mesher/discussion/905307/thread/946ccfc2/), (in file _/usr/share/netgen/drawing.tcl_ change _-indirect true_ to _-indirect false_)

##### Result viewer

    sudo apt-get install paraview

##### Wiki

    sudo apt-get install retext

##### Text processing

    sudo apt-get install texlive texlive-science texmaker
