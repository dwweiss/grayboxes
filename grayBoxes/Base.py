"""
  Copyright (c) 2016- by Dietmar W Weiss

  This is free software; you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as
  published by the Free Software Foundation; either version 3.0 of
  the License, or (at your option) any later version.

  This software is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this software; if not, write to the Free
  Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02110-1301 USA, or see the FSF site: http://www.fsf.org.

  Version:
      2018-06-04 DWW

Note on excution of a python script in from file manager on Windows:
    1. Install Python 3.0 or newer
    2. Find location of "python.exe" employing windows command "where.exe":
       > where python.exe
    3. In file manager, [Properties] -> [Change] -> [Browse] to: 'python.exe'
    4. Now this python script starts in file manager with double-click

Note on program arguments:
    - no arguments                 : program starts in default mode
    - two arguments: 'path inputFile'
                                   : command line mode with password protection
    - three arguments '-s path inputFile'
                                   : command line mode, no password
    - two arguments '-s -g'        : graphic mode, no console output
    - one argument '-g'            : graphic mode with information to console


Example: see code below line starting with: "if __name__ == '__main__':"
"""

from io import IOBase
from datetime import datetime
from getpass import getpass
from hashlib import sha224
from re import sub
from sys import version_info, exit, stdout, argv, modules
from tempfile import gettempdir
from time import time
import numpy as np
try:
    from tkinter import Button
    from tkinter import Entry
    from tkinter import Label
    from tkinter import messagebox
    from tkinter import Tk
except ImportError:
    print("\n!!! Wrong Python interpreter (version: '" +
          str(version_info.major) + '.' + str(version_info.major) +
          '.' + str(version_info.micro) + "') or 'tkinter' not imported")
try:
    import parallel
except ImportError:
    print("!!! Module 'parallel' not imported")


class Base(object):
    """
    Base connects objects in process modeling and controls their execution

    The objects are organized in overlapping tree structures. The concept of
    conservative leader-follower relationships (authoritarian) is extended by
    leader-cooperator relationships (partnership). The differentiation into
    leader-follower and leader-cooperator relationships allows the creation of
    complex object structures which can coexist in space, time or abstract
    contexts.

    The implementation supports:

    - Distributed development of objects derived from the connecting base class

    - Object specialization in "vertical" and "horizontal" direction
        - leader-type objects implementing alternative control of tasks
        - peer-type opjects implement empirical submodels (data-driven),
          theoretical submodels, knowledge objects (e.g. properties of matter)
          and generic service objects (e.g. plotting devices)

     - Uniform interface to code execution of objects derived from the
       connector class objects via public methods
        - pre-process:   pre() which calls load()
        - main task:    task() which is called by control()
        - post-process: post() which calls save()

      The __call__() method calls pre(), control() and post() recursively.
      Iterative or transient repetition of the task() method will be
      controlled in derived classes by a overloaded control(), see class Loop

        leader object
           ^
           |
         --|---------                                 transient or
        | leader()   |                             ---iterative----
        |------------|         ---> pre()         |                ^
        |            |        |                   v                |
        | self.__call__()  ---+---> control() ----o---> task() ----o---> stop
        |            |        |
        |            |         ---> post()
        |            |
        |------------|
        | follower() |
         --|---------
           |-> follower/cooperator.__call__()
           |-> follower/cooperator.__call__()
           :

      The execution sequence in a tree of objects is outlined in the figure
      below: recursive call of pre() for object (1) and its followers (11),
      (12), (111), (112) and its cooperators (21) and (211).

      Execution of methods task() and post() is similar. Note that method pre()
      of object (2) is not executed.

       -----------                                   -----------
      | no leader |                                 | no leader |
      |-----------|                                 |-----------|
      |           |                                 |           |
      |object (1) |                                 |object (2) |
      |   Base -----> pre()                         |    Base   |
      |           |                                 |           |
      |-----------|                                 |-----------|
      | follower()|                                 | follower()|
       --|--------                                   --|----------
         |                                             |
         +----------------------+---------------+      |
         |                      |               |      |
       --|--------            --|--------       |    --|--------
      | leader()  |          | leader()  |      |   | leader()  |
      |-----------|          |-----------|      |   |-----------|
      |           |          |           |      +-->|           |
      |object (11)|          |object (12)|          |object (21)|
      |   Base -----> pre()  |   Base -----> pre()  |   Base -----> pre()
      |           |          |           |          |           |
      |-----------|          |-----------|          |-----------|
      | follower()|          |no follower|          | follower()|
       --|--------            -----------            --|--------
         |                                             |
         +----------------------+                      |
         |                      |                      |
       --|--------            --|--------            --|--------
      | leader()  |          | leader()  |          | leader()  |
      |-----------|          |-----------|          |-----------|
      |           |          |           |          |           |
      |object(111)|          |object(112)|          |object(211)|
      |   Base -----> pre()  |   Base -----> pre()  |   Base -----> pre()
      |           |          |           |          |           |
      |-----------|          |-----------|          |-----------|
      |no follower|          |no follower|          |no follower|
       -----------            -----------           ------------
    """

    def __init__(self, identifier='Base', argv=None):
        self._identifier = str(identifier)
        self.argv = argv
        self.program = self.__class__.__name__
        self.version = '010518_dww'

        self._execTimeStart = 0.0       # start measuring execution time
        self._minExecTimeShown = 1.0    # times less than limit are not shown
        self.path = None                # path to files, see @path.setter
        self.extension = None           # file extension, see @extension.setter

        self._gui = False               # no graphic user interface if False
        self._batch = False             # no user interaction if True
        self._silent = False            # no console output if True
#        self._silentPrevious = False    # previous value of self._silent

        self._ready = True              # if True, successful train/prediction

        self._preDone = False           # internal use: True if  pre() is done
        self._taskDone = False          # internal use: True if task() is done
        self._postDone = False          # internal use: True if post() is done

        self._lineCompleted = True      # helper variable for self.write()
        self._logFile = None            # log file, see self.write()

        self._leader = None             # binding to leader object
        self._followers = []            # array of bindings to followers

        self._data = None               # binding to DataFrame
        self._csvSeparator = ','        # separator in csv-files

    def __call__(self, **kwargs):
        """
        Executes model

        Args:
            kwargs (dict, optional):
                keyword arguments passed to pre(), control() and post()

                silent (bool):
                    if True then suppress printing
        Return:
            (float):
                residuum from range 0.0 .. 1.0 indicating error of task
                or -1.0 if parallel and rank > 0
        """
        # skip model execution if parallelized with MPI and rank > 0
        if parallel.rank():
            return -1.0

        self.silent = kwargs.get('silent', self.silent)

        self.prolog()
        self.pre(**kwargs)

        res = self.control(**kwargs)

        self.post(**kwargs)
        self.epilog()
        return res

    def __str__(self):
        s = ''
        if not self.leader:
            s += "@root: '" + self.identifier + "', \n"
        s += "{identifier: '" + self.identifier + "'"
        if self.leader:
            s += ", level: '" + str(self.treeLevel()) + "'"
            s += ", leader: '" + self.leader.identifier + "'"
            if self.identifier != self.leader:
                s += ' (follower)'
        if self.followers:
            s += ", followers: ["
            for x in self.followers:
                if x:
                    s += "'" + x.identifier + "', "
                else:
                    s += "'None', "
            if self.followers:
                s = s[:-2]
            s += ']'
        s += '}'

        for x in self.followers:
            if x:
                s += ',\n' + x.indent() + str(x)
                # if self.isCooperator(x): s = '# (cooperator)'
        return s

    def destruct(self):
        if self._data:
            del self._data
        if self._logFile:
            self._logFile.close()

        return self.destructDownwards(fromNode=self)

    def destructDownwards(self, fromNode):
        if not fromNode:
            return False
        for i in range(len(fromNode.followers)):
            node = fromNode.followers[i]
            if node:
                if id(node.leader) == id(fromNode):
                    self.destructDownwards(node)
        if fromNode.leader:
            fromNode.leader.destructFollower(fromNode)
        return True

    def destructFollower(self, node):
        if not node:
            return False
        i = -1
        for index, val in enumerate(self.followers):
            if id(val) == id(node):
                i = index
                break
        if i == -1:
            return False
        if self.isCooperator(node):
            return False
        del node._data
        if node._logFile is not None:
            node._logFile.close()
        self._followers[i] = None
        return True

    def root(self):
        """
        Returns:
            (binding):
                root node of this tree (the node which leader is None)
        """
        p = self
        while p.leader:
            p = p.leader
        return p

    def treeLevel(self):
        """
        Returns:
            (int):
                level of this node in tree, relative to root (root is 0)
        """
        n = 0
        p = self.leader
        while p:
            p = p.leader
            n += 1
        return n

    def indent(self):
        return '    ' * self.treeLevel()

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, value):
        if value:
            self._identifier = str(value)
        else:
            self._identifier = self.__class__.__name__

    @property
    def argv(self):
        return self._argv

    @argv.setter
    def argv(self, value):
        if value is None:
            self._argv = argv
        else:
            self._argv = value

    @property
    def gui(self):
        return self._gui

    @gui.setter
    def gui(self, value):
        if 'tkinter' not in modules:
            value = False
            self.write("!!! 'gui' is not set: no module 'tkinter'")
        self._gui = bool(value)
        for x in self._followers:
            x._gui = bool(value)

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, value):
        self._batch = bool(value)
        for x in self._followers:
            x._batch = bool(value)

    @property
    def silent(self):
        return self._silent

    @silent.setter
    def silent(self, value):
        self._silent = value

    @property
    def ready(self):
        return self._ready

    @ready.setter
    def ready(self, value):
        self._ready = value

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if not value:
            self._path = gettempdir()
        else:
            self._path = str(value)
        self._path = self._path.replace('\\', '/')
        if self._path[-1] != '/':
            self._path += '/'

    @property
    def extension(self):
        return str(self._extension)

    @extension.setter
    def extension(self, value):
        if not value:
            self._extension = ''
        else:
            if not value.startswith('.'):
                self._extension = '.' + str(value)
            else:
                self._extension = str(value)

    @property
    def logFile(self):
        return self._logFile

    @logFile.setter
    def logFile(self, file):
        """
        - Closes log file if open.
        - Opens new log file if 'file' is not None. If 'file' == "", the
          identifier of the object is used as log file name.
        - Assigns 'None' to log file if opening of log file fails.

        Args:
            file (str or IOBase or None):
                file name or binding to log file; 'None' stops logging
        """
        if self._logFile is not None:
            self._logFile.close()
            self._logFile = None

        if file is None:
            return

        if isinstance(file, IOBase):
            self._logFile = file
            return

        if not file:
            file = self.identifier
            print('file:', file)
            file += '.log'
            try:
                self._logFile = open(self.path + file, 'w')
                self.write('+++ log: ', "'", file, "'")
            except IOError:
                self._logFile = None
                print("??? self._logFile: '" + self.path + str(file) +
                      "' not open for writing")

    @property
    def csvSeparator(self):
        return str(self._csvSeparator)

    @csvSeparator.setter
    def csvSeparator(self, value):
        if value is None:
            self._csvSeparator = ' '
        else:
            self._csvSeparator = value

    @property
    def leader(self):
        return self._leader

    @leader.setter
    def leader(self, other):
        if other:
            other.setFollower(self)

    @property
    def followers(self):
        return self._followers

    @followers.setter
    def followers(self, other):
        self.terminate("followers.setter: 'followers' is protected," +
                       " use 'setFollower()'")

    def isRoot(self):
        return not self.leader

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, other):
        if self._data is not None:
            if not self.silent:
                print("+++ data.setter: delete 'data'")
            del self._data
        # if not self.silent: print("+++ data.setter: assign new 'data'")
        self._data = other

    def setFollower(self, other):
        if other:
            if not isinstance(other, (list, tuple)):
                other._leader = self
                if other not in self._followers:
                    self._followers.append(other)
            else:
                for o in other:
                    o._leader = self
                    if o not in self._followers:
                        self._followers.append(o)
        return other

    def getFollower(self, identifier):
        """
        Search for node with given 'identifier'. Search starts downwards from
        root

        Args:
            identifier (str):
                identifier of searched node

        Returns:
            (binding):
                node with given identifier
        """
        return self.getFollowerDownwards(identifier, fromNode=None)

    def getFollowerDownwards(self, identifier, fromNode=None):
        """
        Search for node with given 'identifier', start search downwards
        from 'fromNode'

        Args:
            identifier (str):
                identifier of wanted node

            fromNode (binding):
                start node for downward search. If 'fromNode' is None, search
                starts from root

        Returns:
            (binding to tree node)
                node with given identifier
        """
        if self.identifier == identifier:
            return self
        if fromNode:
            if fromNode._identifier == identifier:
                return fromNode
        else:
            fromNode = self.root()
            if not fromNode:
                return None
        if fromNode.identifier == identifier:
            return fromNode
        for i in range(len(fromNode.followers)):
            node = fromNode.followers[i]
            if node:
                node = self.getFollowerDownwards(identifier, node)
                if node:
                    return node
        return None

    def isFollower(self, other):
        return other._leader == self and other in self._followers

    def setCooperator(self, other):
        if other:
            if not isinstance(other, (list, tuple)):
                if other not in self._followers:
                    self._followers.append(other)
            else:
                for o in other:
                    if o not in self._followers:
                        self._followers.append(o)
        return other

    def isCooperator(self, other):
        return other._leader != self and other in self._followers

    def cleanString(self, s):
        return sub('[ \t\n\v\f\r]', '', s)

    def reverseString(self, s):
        rs = list(s)
        rs.reverse()
        return ''.join(rs)

    def kwargsDel(self, kwargs, remove):
        """
        Makes copy of keyword dictionary and removes given keys

        Args:
            kwargs (dict):
                keyword arguments

            remove (str or list of str):
                keywords of items to be removed

        Returns:
            dict of keyword arguments without removed items
        """
        dic = kwargs.copy()
        for key in np.atleast_1d(remove):
            if key in dic:
                del dic[key]
        return dic

    def kwargsGet(self, kwargs, keys, default=None):
        """
        Returns value of kwargs for first matching key, otherwise  return
        'default' value

        Args:
            kwargs (dict):
                keyword arguments

            keys (str or array_like of str):
                keyword or list of alternative keywords

            default(any type, optional):
                value to be returns if none of the keys in kwargs

        Returns:
            (any type):
                value of first matching key or value of 'default'
        """
        for key in np.atleast_1d(keys):
            if key in kwargs:
                return kwargs[key]
        return default

    def terminate(self, message=''):
        if not message:
            message = 'Fatal error'
        self.write("???\n??? '" + self.program + "', terminated due to: '" +
                   message + "'\n???")
        if self.gui:
            messagebox.showerror("Termination of: '" + self.program + "'",
                                 message)
        exit()

    def warning(self, message='', wait=False):
        if not message:
            message = 'Warning'
        wait = bool(wait)
        self.write("!!! '", self.program, "', warning: '", message, "'")
        if self.gui:
            messagebox.showinfo(self.program + ' - Warning', message)
        else:
            if not self.silent and wait:
                # os.system('pause')
                input('!!! Press Enter to continue ...')

    def write(self, *message):
        """
        Sends output to console (if not in silent mode) and to log file of root

        Wraps print() function considering indent of follower objects

        Args:
            message (list):
                list of variable arguments to be written; if last argument is
                'None', line wont be completed (no carriage return)
        Note:
            No seperator between print elements, comparable to print(x, sep='')
        """
        silent = self.silent or parallel.rank()

        lf = self._logFile
        if lf is None:
            lf = self.root()._logFile

        if self._lineCompleted:
            if not silent:
                print(self.indent(), end='')
            if lf:
                print(self.indent(), end='', file=lf)
        for x in message:
            if x is not None:
                if not silent:
                    print(str(x), end='', sep='')
                if lf:
                    print(str(x), end='', sep='', file=lf)
        if not message or message[-1] is not None:
            if not silent:
                print()
            if lf:
                print(file=lf)
            self._lineCompleted = True

    def _qualify(self):
        """
        Asks for password. Terminates program if password is wrong

        Note:
            Create new hash string 's' with:
                import hashlib
                s = hashlib.sha224('new password'.encode('UTF-8')).hexdigest()
        """
        s = 'c0dad715ce5501ea5e382d3a44a7cf816f9a1a309dfeb88cbe9ebfbd'
        if self.gui:
            parent = Tk()
            parent.title(self.program)
            Label(parent, text='').grid(row=0, column=0)
            Label(parent, text='Enter password').grid(row=1, column=0)
            Label(parent, text='').grid(row=2, column=0)
            entry2 = Entry(parent, show='*')
            entry2.grid(row=1, column=1)
            Button(parent, text='Continue',
                   command=parent.quit).grid(row=3, column=5, pady=10)
            parent.mainloop()
            pw = entry2.get()
            parent.withdraw()
        else:
            stdout.flush()
            pw = getpass('Enter password: ')
        if sha224(pw.encode('UTF-8')).hexdigest() != s:
            self.terminate('wrong password')

    def prolog(self, purpose='Processing data',
               usage='[ path sourceFile [ -s -g ] ]',
               example='-g -s d:/path/ test.xml'):
        if '-h' in self.argv or '--help' in self.argv:
            print("This is: '" + self.program + "', version " + self.version)
            print('\nPurpose: ' + purpose)
            print('Usage:   ' + self.program + ' ' + usage)
            print('Example: ' + self.program + ' ' + example)
            exit()

        qualify = False
        if len(self.argv) > 1+0:
            self.gui = '-g' in self.argv or '--gui' in self.argv
            self.silent = '-s' in self.argv or '--silent' in self.argv
            if not self.gui and self.silent:
                qualify = False
        else:
            if not self.gui:
                # self.silent = False
                pass

        if self.isRoot():
            if self._logFile is None:
                self.logFile = None  # see setter, assigns: identifier + '.log'

        if self.isRoot():
            self.write('=== path: ', "'", self.path, "'")
            silentPrevious = self.silent
            self.silent = True
            self.write('+++ date: ', str(datetime.now().date()),
                       ' ', str(datetime.now().time())[:8])
            self.silent = silentPrevious

            self.write()
            self.write("*** This is: '", self.program, "'", None)
            if self.identifier and self.identifier != self.program:
                self.write(", id: '", self.identifier, "'", None)
            self.write(", version: '", self.version, "'")
            if qualify:
                self._qualify()
                pass

        if self.isRoot():
            self.write('=== Pre-processing')
            self._execTimeStart = time()

    def epilog(self):
        if self.isRoot():
            message = "'" + self.program + "' is successfully completed."
            execTime = time() - self._execTimeStart
            if execTime >= self._minExecTimeShown:
                self.write('    Execution time: ', format(round(execTime, 2)))
            self.write('*** ', message, '\n')

        if self._logFile is not None:
            self._logFile.close()
            self._logFile = None

        for x in self.followers:
            x.epilog()

        if self.isRoot() and self.gui:
            messagebox.showinfo(self.program, message)

    def load(self):
        return True

    def save(self):
        return True

    def initialCondition(self):
        # super().initialCondition()                # use it in derived classes
        pass

    def updateNonLinear(self):
        # super().updateNonLinear()                 # use it in derived classes
        pass

    def updateTransient(self):
        # super().updateTransient()                 # use it in derived classes
        pass

    def pre(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments

        Returns:
            (bool):
                False if data loading failed
        """
        ok = True
        for x in self.followers:
            x.pre(**kwargs)
            if self.isCooperator(x):
                self.write(self.indent() + "    ['" + x.identifier +
                           "' is cooperator]")
        if self.root().followers:
            self.write('--- Pre (', self.identifier, ')')

        if self.data is None:
            ok = self.load()
        self._preDone = True
        return ok

    def task(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments

        Returns:
            (float):
                residuum from range [0., 1.], indicating error
        """
        for x in self.followers:
            x.task(**kwargs)
            if self.isCooperator(x):
                self.write(self.indent() + "    ['" + x.identifier +
                           "' is cooperator]")
        if self.root().followers:
            self.write('--- Task (', self.identifier, ')')
        self._taskDone = True
        return 0.0

    def post(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments

        Returns:
            (bool):
                if False, data saving failed
        """
        ok = True
        for x in self.followers:
            x.post(**kwargs)
            if self.isCooperator(x):
                self.write(self.indent() + "    ['" + x.identifier +
                           "' is cooperator]")
        if self.root().followers:
            self.write('--- Post (', self.identifier, ')')
        if self.data is None:
            ok = self.save()
        self._postDone = True
        return ok

    def control(self, **kwargs):
        """
        Args:
            kwargs (dict, optional):
                keyword arguments passed to task()

        Returns:
            (float):
                residuum from range [0., 1.], indicating error
        """
        if self.isRoot():
            execTime = time() - self._execTimeStart
            if execTime >= self._minExecTimeShown:
                self.write('    Execution time: {:2f} s'.format(round(execTime,
                                                                      2)))
            self._execTimeStart = time()

        if self.isRoot():
            self.write('=== Task-processing')
        res = self.task(**kwargs)

        if self.isRoot():
            execTime = time() - self._execTimeStart
            if execTime >= self._minExecTimeShown:
                self.write('    Execution time: {:2f} s'.format(round(execTime,
                                                                      2)))
            self._execTimeStart = time()
        self.write('=== Post-processing')
        return res


# Examples ####################################################################

if __name__ == '__main__':
    ALL = 1

    if 0 or ALL:
        class Foo(Base):
            def __init__(self, identifier=''):
                super().__init__(identifier=identifier)
                self.x = 3.0

            def task(self, **kwargs):
                super().task()

                self.x *= 2.0
                b = 7.0
                for i in range(int(1e6)):
                    b = b / 3.0
                self.y = b
                return 0.0

            def post(self, **kwargs):
                super().post()
                self.write('    x: ', self.x)
                self.write('    y: ', self.y)

        # creates instance
        foo = Foo('root')
        foo.gui = False

        # assigns path to files
        foo.path = 'c:/Temp/'

        # creates objects
        f1 = Foo('follower 1')
        f11 = Foo('follower 1->1')
        f12 = Foo('follower 1->2')
        f13 = Foo('follower 1->3')
        f2 = Foo('follower 2')
        f21 = Foo('follower 2->1 and cooperator 1--2')
        f22 = Foo('follower 2->2')

        # connects objects                                    foo
        foo.setFollower([f1, f2])           # .             /     \
        f1.setFollower([f11, f12, f13])     # .          f1 ......   f2
        f2.setFollower([f21, f22])          # .        / |  \     : /  \
        f1.setCooperator(f21)               # .     f11 f12 f13   f21 f22

        # links between two objects
        f13.x = 6.789
        print('f13.x:', f13.x)
        f1.link = foo.getFollower('follower 1->3')
        f1.link.x = 4.56
        print('f1.link.id:', f1.link.identifier)
        print('f13.id:', f13.identifier)
        print('f13.x:', f13.x)

        # assigns a private logfile to follower f11
        f11.logFile = 'f11log'
        f11.write('abc')
        foo()

        # prints content of root and its followers
        print('\nPrint(foo): ' + str(foo))

        if 0 or ALL:
            # assigns a new log file common to root and its followers
            foo.logFile = 'abc'
            foo()

        if 0 or ALL:
            # searches for specific follower in tree
            identifier = 'follower 11'
            p = foo.getFollowerDownwards(identifier=identifier)
            if p is None:
                print('identifier not found, p:', p)
            else:
                print('identier found:', p.identifier == identifier)
                print('downward search, p.identifier:', p.identifier)

        if 0 or ALL:
            # destructs tree
            print('*** destruct')
            print('foo 1:', foo)
            foo.destruct()

        if 0 or ALL:
            # sends warning and termination of program
            print('foo 2:', foo)
            foo.warning('my warning1')
            foo.terminate('my error message')
