### Raspberry Pi installation proposal

<!-- Version: 2018-08-11 DWW -->

In the text below **USER** depicts the actual user name *pi* and **X** the disk partition for the workspace

##### System

- Download *balenaElcher* on your PC
- Download a *YEAR-MM-DD-raspbian-buster-zip* distribution
- Create a mcro SD card (>= 16 GB, class 10)
- Start linux on the Raspberry Pi

- Consider a directory **/X** besides the default /home/**USER** directory
 
      sudo chown USER /X
      ln -s /X /home/USER/X

- Update Linux

      sudo apt update && sudo apt -y upgrade

- Install Beryconda

      Download the following script, make it executable and execute it
      https://github.com/jjhelmus/berryconda/releases/download/v2.0.0/Berryconda3-2.0.0-Linux-armv7l.sh

- Optional: Folder encryption

      sudo add-apt-repository ppa:gencfsm && sudo apt update && sudo apt -y install ecryptfs-utils gnome-encfs-manager

- Optional: Monitor hardware

      sudo apt install cpufrequtils
      sudo apt install conky

- Basic tools

      sudo apt -y install cmake unzip pkg-config git 
    
##### Seabreeze

      conda install -c poehlmann python-seabreeze 
      
##### Remote connection        

    http://www.circuitbasics.com/how-to-connect-to-a-raspberry-pi-directly-with-an-ethernet-cable/
    
##### Python 3

    python3 -V
    sudo apt -y install python3-pip python-pip build-essential libssl-dev libffi-dev python3-dev python3-tk
    sudo apt -y install libopenblas-dev liblapack-dev libhdf5-serial-dev python-h5py python-yaml
    
    pip3 install --user futures traitsui openpyxl xlrd numba
    
##### Spyder (python development)

    pip3 install --user rope && sudo apt -y install spyder3
    
##### Genetic algorithms

    pip3 install --user modestga

##### Graphics

    sudo apt -y install vtk6 python-vtk6 && pip3 install --user mayavi
    sudo apt -y install mayavi2
    
##### GUI

    sudo apt -y install python-wxtools python-traitsui               # loop in wx
    
    # Qt-designer as alternative to TraitsUi: 
    sudo apt -y install python-qt4 qt4-designer        # only if GUI design as *.ui file

##### Text processing

    sudo apt -y install texlive texlive-science texmaker

##### Administration

    sudo apt -y install fslint                          # find duplicate files
    

    

