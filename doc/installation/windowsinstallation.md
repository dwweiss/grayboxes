### Installation of downloaded python packages on windows

<!-- Version: 2018-06-21 DWW -->

In the text below **PACKAGE** the python package name and **EXEPATH** the location of _python*.exe_ 

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

