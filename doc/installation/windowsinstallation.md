### Installation of python packages on windows with pip

In this section **PACKAGE** depicts the python package name and **EXEPATH** the location of _pip*.exe_ 

1. Open console
2. Find location of _pip*.exe_
    
          where /R c:\ pip*.exe
   
   Note this path (e.g. _c:\ProgramData\Anaconda3_) as **EXEPATH** 
   
3. Install with 

          EXEPATH\pip.exe install PACKAGE


### Installation of downloaded python packages on windows

In this section **PACKAGE** depicts the python package name and **EXEPATH** the location of _python*.exe_ 

1. Open console
2. Find location of _python*.exe_
    
          where /R c:\ python*.exe
   
   Note this path (e.g. _c:\ProgramData\Anaconda3_) as **EXEPATH** 
   
3. Download package as **PACKAGE**.zip file
4. Unpack **PACKAGE**.zip file
5. Change in console to the directory containing the unzipped **PACKAGE** 
6. Check if _setup.py_ does exist
7. Install with 

          EXEPATH\python.exe setup.py install

