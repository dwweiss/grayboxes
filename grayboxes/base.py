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
      2019-12-06 DWW

  Note on program arguments:
    - no arguments          : program starts in default mode
    - two arguments: 'path input_file'
                            : command line mode with password protection
    - three arguments '-s path input_file' or
                      '--silent path input_file'
                            : command line mode, no password
    - two arguments '-s -g' : graphic mode, no console output or
                    '--silent', '--gui'
    - one argument '-g'     : graphic mode with information to console
                    '--gui'

"""

__all__ = ['Base', 'Float1D', 'Float2D', 'Float3D', 'Str1D', 'Function']

import os
from datetime import datetime
from getpass import getpass
from hashlib import sha224
import collections
from matplotlib.figure import Figure
#from nptyping.array import Array
import numpy as np
from path import Path
from re import sub
import sys
from time import time
from tempfile import gettempdir
from typing import (Any, Callable, Dict, Iterable, Sequence, Tuple, List, 
                    Optional, Union)
try:
    from tkinter import Button
    from tkinter import Entry
    from tkinter import Label
    from tkinter import messagebox
    from tkinter import Tk
except ImportError:
    print("\n!!! Wrong Python interpreter (version: '" +
          str(sys.version_info.major) + '.' + str(sys.version_info.major) +
          '.' + str(sys.version_info.micro) + "') or 'tkinter' not imported")
import logging
logger = logging.getLogger(__name__)

try:
    import grayboxes.parallel as parallel
except ImportError:
    try:
        import parallel as parallel
    except ImportError:
        print('!!! Module parallel not imported')

#Float1D  = Optional[Array[float, ...]]
#Float2D  = Optional[Array[float, ..., ...]]
#Float3D  = Optional[Array[float, ..., ..., ...]]
Float1D  = Optional[np.ndarray]
Float2D  = Optional[np.ndarray]
Float3D  = Optional[np.ndarray]
Str1D    = Optional[Iterable[str]]
Function = Optional[Callable[..., List[float]]]


class Base(object):
    """
    Connects model objects and controls their execution

    The objects are organized in overlapping tree structures. The
    concept of conservative leader-follower relationships
    (authoritarian) is extended by leader-cooperator relationships
    (partnership). The differentiation into leader-follower and
    leader-cooperator relationships allows the creation of complex
    object structures which can coexist in space, time or abstract
    contexts.

    The implementation supports:

    - Distributed development of objects derived from the connecting
      Base class

    - Object specialization in "vertical" and "horizontal" direction
        - leader-type objects implementing alternative control of tasks
        - peer-type objects implement empirical submodels (data-driven),
          theoretical submodels, knowledge objects (e.g. properties of
          matter) and generic service objects (e.g. plotting)

     - Uniform interface to code execution of objects derived from the
       connector class objects via public methods
        - pre-process:   pre() which calls load()
        - main task:    task() which is called by control()
        - post-process: post() which calls save()

      The __call__() method calls pre(), control() and post()
      recursively. Iterative or transient repetition of the task()
      method will be controlled in derived classes by a overloaded
      control(), see class Loop


    leader object
       ^
       |
     --|---------                                transient or
    | leader()   |                            ---iterative----
    |------------|        ---> pre()         |                ^
    |            |       |                   v                |
    | self.__call__() ---+---> control() ----o---> task() ----o---> stop
    |            |       |
    |            |        ---> post()
    |            |
    |------------|
    | follower() |
    --|---------
      |-> follower/cooperator.__call__()
      |-> follower/cooperator.__call__()
      :

      The execution sequence in a tree of objects is outlined in the
      figure below: recursive call of pre() for object (1) and its
      followers (11), (12), (111), (112) and its cooperators (21) and
      (211).

      Execution of methods task() and post() is similar. Note that
      method pre() of object (2) is not executed.

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
         +----------------------+---------------       |
         |                      |               |      |
       --|--------            --|--------       |    --|--------
      | leader()  |          | leader()  |      |   | leader()  |
      |-----------|          |-----------|      |   |-----------|
      |           |          |           |       -->|           |
      |object (11)|          |object (12)|          |object (21)|
      |   Base -----> pre()  |   Base -----> pre()  |   Base ----> pre()
      |           |          |           |          |           |
      |-----------|          |-----------|          |-----------|
      | follower()|          |no follower|          | follower()|
       --|--------            -----------            --|--------
         |                                             |
         +----------------------                       |
         |                      |                      |
       --|--------            --|--------            --|--------
      | leader()  |          | leader()  |          | leader()  |
      |-----------|          |-----------|          |-----------|
      |           |          |           |          |           |
      |object(111)|          |object(112)|          |object(211)|
      |   Base -----> pre()  |   Base -----> pre()  |   Base ----> pre()
      |           |          |           |          |           |
      |-----------|          |-----------|          |-----------|
      |no follower|          |no follower|          |no follower|
       -----------            -----------           ------------
    """

    def __init__(self, identifier: str = 'Base',
                 argv: Optional[List[str]] = None) -> None:
        """
        Initializes object

        Args:
            identifier:
                Unique identifier of object

            argv:
                Program arguments
        """
        if not identifier:
            identifier = self.__class__.__name__
        self._identifier: str = str(identifier)
        self.argv: Optional[List[str]] = argv
        if argv is None:
            self.argv = sys.argv
        self.program: str = self.__class__.__name__
        self.version: str = '19.11'

        self.path: Optional[Path] = None       # path to file,see setter
        self.extension: str = '.data'          # file ext., see setter

        self._exe_time_start: float = 0.0      # start measure exec.time
        self._min_exe_time_shown: float = 1.0  # times < limit not shown

        self._gui: bool = False                # graph.interface if True
        self._batch: bool = False              # user interact. if False
        self._silent: bool = False             # console output if False

        self._ready: bool = True               # success of train/pred

        self._pre_done: bool = False           # True if  pre() done
        self._task_done: bool = False          # True if task() done
        self._post_done: bool = False          # True if post() done

        self._leader: Optional['Base'] = None  # leader object
        self._followers: List[Optional['Base']] = []
                                               # follower list
        self._links: List[Optional['Base']] = []
                                               # link list 

        self._data: Optional[Any] = None       # data sets
        self._csv_separator: str = ','         # separator in csv-files

        # figure requires subplots, eg axes = self.figure.suplots(2)
        #                              axes[0].plot([1.,2.], [4.5, 9.1])
        self._figure: Optional[Figure] = Figure()
        
    def __call__(self, **kwargs: Any) \
            -> Union[float,              
                     Dict[str, Any],   
                     Float2D,          
                     Tuple[Float1D, Float1D], 
                     Tuple[Float1D, Float2D], 
                     Tuple[Float2D, Float2D]]: 
        """
        Executes object

        Kwargs:
            silent (bool):
                if True, then suppress printing

            Keyword arguments to be passed to self.control()

        Returns:
            see self.control()
        """
        # skip model execution if parallelized with MPI and rank > 0
        if 'parallel' in sys.modules and parallel.rank():
            return -1.0

        if 'silent' in kwargs:
            self.silent = kwargs['silent']

        ok = self.prolog()
        if not ok:
            self.write('??? Base.prolog() returned with False\n')
       
        ok = self.pre(**kwargs)
        if not ok:
            self.write('??? Base.pre() returned with False\n')

        task_result: Union[float, 
                           Dict[str, Any],
                           Float2D,
                           Tuple[Float2D, Float2D]] = self.control(**kwargs)

        ok = self.post(**kwargs)
        if not ok:
            self.write('??? Base.post() returned with False\n')
            
        ok = self.epilog()
        if not ok:
            self.write('??? Base.epilog() returned with False\n')

        return task_result

    def __str__(self) -> str:
        s = ''
        if not self.leader:
            s += "@root: '" + self.identifier + "', \n"
        s += "{identifier: '" + self.identifier + "'"
        if self.leader:
            s += ", level: '" + str(self.tree_level()) + "'"
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
        if self.links:
            s += ", links: ["
            for x in self.links:
                if x:
                    s += "'" + x.identifier + "', "
                else:
                    s += "'None', "
            if self.links:
                s = s[:-2]
            s += ']'
        s += '}'

        for x in self.followers:
            if x:
                s += ',\n' + x.indent() + str(x)
                # if self.isCooperator(x): s = '# (cooperator)'
        return s

    def destruct(self) -> bool:
        """
        Destructs all followers. Cooperators will be kept

        Returns:
            True on success
        """
        if self._data:
            del self._data
        if self.is_root():
            logging.shutdown()

        return self.destruct_downwards(from_node=self)

    def destruct_downwards(self, from_node: 'Base') -> bool:
        """
        Destructs all followers downwards from 'from_node'.
        Cooperators will be kept

        Args:
            from_node:
                start node of search

        Returns:
            True on success
        """
        if not from_node:
            return False
        for i in range(len(from_node.followers)):
            node = from_node.followers[i]
            if node:
                if id(node.leader) == id(from_node):
                    self.destruct_downwards(node)
        if from_node.leader:
            from_node.leader._destruct_follower(from_node)
            
        return True

    def _destruct_follower(self, node: 'Base') -> bool:
        """
        Destructs the followers of 'node'. Cooperators will be kept

        Args:
            node:
                actual node

        Returns:
            False if this node has not followers
        """
        if not node:
            return False
        i = -1
        for index, val in enumerate(self.followers):
            if id(val) == id(node):
                i = index
                break
        if i == -1:
            return False
        if self.is_cooperator(node):
            return False
        del node._data
        self._followers[i] = None
        
        return True

    def is_root(self) -> bool:
        return not self.leader

    def root(self) -> 'Base':
        """
        Returns:
            Root node of this tree (leader of root node is None)
        """
        p = self
        while p.leader:
            p = p.leader
        return p

    def tree_level(self) -> int:
        """
        Returns:
            Level of this node in tree, relative to root (root is 0)
        """
        n = 0
        p = self.leader
        while p:
            p = p.leader
            n += 1
        return n

    def indent(self) -> str:
        return (4 * ' ') * self.tree_level()

    @property
    def identifier(self) -> str:
        return self._identifier

    @identifier.setter
    def identifier(self, value: str) -> None:
        if value:
            self._identifier = str(value)
        else:
            self._identifier = self.__class__.__name__

    @property
    def argv(self) -> List[str]:
        return self._argv

    @argv.setter
    def argv(self, value: Optional[Sequence[str]]) -> None:
        if value is None:
            self._argv = sys.argv
        else:
            self._argv = list(value)

    @property
    def gui(self) -> bool:
        return self._gui

    @gui.setter
    def gui(self, value: bool) -> None:
        if 'tkinter' not in sys.modules:
            value = False
            self.warn("!!! 'gui' is not set: module 'tkinter' not imported")
        self._gui = value
        for node in self._followers:
            if node:
                node._gui = value

    @property
    def figure(self) -> Optional[Figure]:
        return self._figure

    @figure.setter
    def figure(self, value: Optional[Figure]) -> None:
        self._figure = value
        for node in self._followers:
            if node:
                node._figure = value

    @property
    def batch(self) -> bool:
        return self._batch

    @batch.setter
    def batch(self, value: bool) -> None:
        self._batch = value
        for node in self._followers:
            if node:
                node._batch = value

    @property
    def silent(self) -> bool:
        return self._silent or \
            bool('parallel' in sys.modules and parallel.rank())

    @silent.setter
    def silent(self, value: bool) -> None:
        self._silent = value
        for node in self.followers:
            if node:
                node.silent = value

    @property
    def ready(self) -> bool:
        return self._ready

    @ready.setter
    def ready(self, value: bool) -> None:
        self._ready = value

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, value: Optional[Union[str, Path]]) -> None:
        if not value:
            p = gettempdir()
        else:
            p = Path(str(value))
        self._path = p 

    @property
    def extension(self) -> str:
        return str(self._extension)

    @extension.setter
    def extension(self, value: Optional[str]) -> None:
        if not value:
            self._extension = ''
        else:
            if not value.startswith('.'):
                self._extension = '.' + str(value)
            else:
                self._extension = str(value)

    @property
    def csv_separator(self) -> str:
        return str(self._csv_separator)

    @csv_separator.setter
    def csv_separator(self, value: Optional[str]) -> None:
        if value is None:
            self._csv_separator = ' '
        else:
            self._csv_separator = value

    @property
    def leader(self) -> Optional['Base']:
        return self._leader

    @leader.setter
    def leader(self, other: Optional['Base']) -> None:
        if other:
            other.set_follower(self)

    @property
    def followers(self) -> List[Optional['Base']]:
        return self._followers

    @followers.setter
    def followers(self, other: Union[Optional['Base'], 
                                     Sequence[Optional['Base']]]) -> None:
        self.set_follower(other)

    @property
    def links(self) -> List[Optional['Base']]:
        return self._links

    @links.setter
    def links(self, other: Union[Optional['Base'], 
                                 Sequence[Optional['Base']]]) -> None:
        self.set_link(other)

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, other: Optional[Any]) -> None:
        if self._data is not None:
            if not self.silent:
                print("+++ data.setter: delete 'data'")
            del self._data
        self._data = other

    def __getitem__(self, identifier: str) -> Optional['Base']:
        """
        Indexing, eg b = Base(); b.followers = ('f1', 'f2'); f1 = b['f1']

        Searches for node with 'identifier'. Starts downwards from root
        If node is not in tree of followers, search will be continued in 
        list of links

        Args:
            identifier:
                Identifier of searched node

        Returns:
            Node with given identifier 
            or 
            None if node not found
        """
        node = self.get_follower(identifier)
        if node is None:
            node = self.get_link(identifier)
        
        return node

    def get_follower(self, identifier: str) -> Optional['Base']:
        """
        Search for node with 'identifier' starts downwards from root

        Args:
            identifier:
                Identifier of searched node

        Returns:
            Node with given identifier 
            or 
            None if node not found
        """
        return self.get_follower_downwards(identifier)

    def get_follower_downwards(self, identifier: str, from_node:
                               Optional['Base'] = None) -> Optional['Base']:
        """
        Search for node with given 'identifier', start search downwards
        from 'from_node'

        Args:
            identifier:
                Identifier of wanted node

            from_node:
                Start node for downward search. If 'from_node' is None,
                search starts from root

        Returns:
            Node with given identifier 
            or 
            None if node not found
        """
        if self.identifier == identifier:
            return self
        if from_node:
            if from_node._identifier == identifier:
                return from_node
        else:
            from_node = self.root()
            if not from_node:
                return None
        if from_node.identifier == identifier:
            return from_node
        for i in range(len(from_node.followers)):
            node = from_node.followers[i]
            if node:
                node = self.get_follower_downwards(identifier, node)
                if node:
                    return node
        return None

    def set_follower(self, other: Union[Optional['Base'], \
                                        Sequence[Optional['Base']]])\
                               -> Union[Optional['Base'], 
                                        Sequence[Optional['Base']]]:
        """
        Adds other node(s)

        Args:
            other:
                Other node or sequence of other nodes

        Returns:
            Reference to 'other'
            
        Example:
            b = Base()
            b.set_follower(Base('follower1'))
            b.set_follower([Base('follower2'), Base('follower3')])
            
            node = b.get_follower('follower2')
            assert node.identifier == 'follower2'
            assert node == b['follower2']
            
            node2 = b['follower2']
            assert node == node2
        """
        if other:
            if not isinstance(other, collections.Sequence):
                other._leader = self
                if other not in self._followers:
                    self._followers.append(other)
            else:
                for node in other:
                    if node:
                        node._leader = self
                        if node not in self._followers:
                            self._followers.append(node)
        return other

    def set_link(self, other: Union[Optional['Base'], \
                                    Sequence[Optional['Base']]])\
                           -> Union[Optional['Base'], 
                                    Sequence[Optional['Base']]]:
        """
        Adds other node(s) to array of links

        Args:
            other:
                Other node or sequence of other nodes

        Returns:
            Reference to 'other'            

        Example:
            b1 = Base()
            b1.set_follower([Base('follower11'), Base('follower12')])
            
            b2 = Base()
            b2.set_link(b1['follower12'])

            assert b2.get_link('follower12') == b2['follower12']
        """
        if other:
            if not isinstance(other, collections.Sequence):
                if other not in self._links:
                    self._links.append(other)
            else:
                for node in other:
                    if node and node not in self._links:
                        self._links.append(node)
        return other

    def get_link(self, identifier: str) -> Optional['Base']:
        """
        Search for node with 'identifier' in list of links

        Args:
            identifier:
                Identifier of searched node

        Returns:
            Node with given identifier 
            or 
            None if node not found
        """
        node = None
        for node in self.links:
            if node:
                if node.identifier == identifier:
                    return node

        return None

    def is_follower(self, other: Optional['Base']) -> bool:
        """
        Args:
            other:
                Other node

        Returns:
            True if 'other' is follower and has 'self' as leader
        """
        if other is None:
            return False
        return other._leader == self and other in self._followers

    def set_cooperator(self, other: Union[Optional['Base'], \
                                          Sequence[Optional['Base']]])\
                                 -> Union[Optional['Base'], 
                                          Sequence[Optional['Base']]]:
        """
        Adds other node as cooperator.
        'other' keep(s) its/their original leader(s)

        Args:
            other:
                Other node or list of other nodes

        Returns:
            'other'
        """
        if other:
            if not isinstance(other, collections.Sequence):
                if other not in self._followers:
                    self._followers.append(other)
            else:
                for node in other:
                    if node:
                        if node not in self._followers:
                            self._followers.append(node)
        return other

    def is_cooperator(self, other: Optional['Base']) -> bool:
        """
        Args:
            other:
                Other node
        
        Returns:
            True if 'other' is follower and has another leader 
            than 'self'
        """
        if not other:
            return False
        return other._leader != self and other in self._followers

    def clean_string(self, s: str) -> str:
        """
        Args:
            s:
                string containing control characters
        
        Returns:
            copy of string 's' without control characters
        """
        return sub('[ \t\n\v\f\r]', '', s)

    def kwargs_del(self, kwargs_: Dict[str, Any],
                   remove: Union[str, Sequence[str]]) -> Dict[str, Any]:
        """
        Makes copy of keyword dictionary and removes given key(s)

        Args:
            kwargs_:
                Dictionary with keywords

            remove:
                Keyword(s) of items to be removed

        Returns:
            Copy of dictionary exclusive removed items
        """
        dic = kwargs_.copy()
        for key in np.atleast_1d(remove):
            if key in dic:
                del dic[key]
        return dic

    def kwargs_get(self, kwargs_: Any,
                   keys: Union[str, Sequence[str]], 
                   default: Any = None) -> Any:
        """
        Returns value of _kwargs for first matching key or 'default' if
        all keys are invalid

        Args:
            kwargs_:
                Dictionay with keyword arguments

            keys:
                Keyword or list of alternative keywords

            default:
                Value to be returned if none of the keys is in '_kwargs'

        Returns:
            Value of first matching key or 'default'
        """
        for key in np.atleast_1d(keys):
            if key in kwargs_:
                return kwargs_[key]
        return default

    def terminate(self, message: str = '') -> None:
        if not message:
            message = 'Fatal error'

        if not self.silent:
            print("\n???\n??? '" + self.program + "', terminated due to: '" +
                  message + "'\n???")
        if self.gui:
            messagebox.showerror("Termination: '" + self.program + "'",
                                 message)
        logger.critical(self.identifier + ' : ' + message)
        self.destruct()

        sys.exit()

    def warn(self, message: str = '', wait: bool = False) -> None:
        """
        - Message to logger
        - Message to TKinter widget if self.gui, otherwise to console

        Args:
            message:
                Warning to be written to log file and console

            wait:
                Wait with program execution if True
        """
        if not self.silent:
            print("!!! '" + self.program + "', warning: '" + message + "'")
        if self.gui:
            messagebox.showinfo(self.program + ' - Warning', message)
        logger.warning(self.identifier + ' : ' + message)
        if not self.silent and wait:
            # consider to replace input() with os.system('pause')
            input('!!! Press Enter to continue ...')

    def write(self, message: str) -> None:
        """
        - Message to logger with file handler
        - Message to console if not in silent mode

        Args:
            message:
                Message to be written to log file and console
        """
        now = datetime.now().strftime('%H:%M:%S.%f')[:-4]
        if not self.silent:
            print(self.indent() + message)
        logger.info(now + ' ' + self.indent() + message)

    def _authenticate(self) -> None:
        """
        Asks for password. Terminates program if wrong password

        Note:
            Create new hash string 's' with:
                import hashlib
                s = hashlib.sha224(
                    'new password'.encode('UTF-8')).hexdigest()
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
            sys.stdout.flush()
            pw = getpass('Enter password: ')
        if sha224(pw.encode('UTF-8')).hexdigest() != s:
            self.terminate('wrong password')

    def prolog(self, purpose: str = 'Processing data',
               usage: str = '[ path sourceFile [ --silent --gui ] | --help ]',
               example: str = '-g -s /tmp test.xml') -> bool:
        if '-h' in self.argv or '--help' in self.argv:
            print("This is: '" + self.program + "', version " + self.version)
            print('\nPurpose: ' + purpose)
            print('Usage:   ' + self.program + ' ' + usage)
            print('Example: ' + self.program + ' ' + example)
            exit()

        ok = True
        authenticate = False
        if len(self.argv) > 1+0:
            self.gui = '-g' in self.argv or '--gui' in self.argv
            self.silent = '-s' in self.argv or '--silent' in self.argv
            if not self.gui and self.silent:
                authenticate = False
        else:
            if not self.gui:
                # TODO self.silent = False
                pass

        global logger
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            f = os.path.join(self.path, self.identifier + '.log')
            handler = logging.FileHandler(f, mode='w')
            handler.setLevel(logging.DEBUG)
            logger.addHandler(handler)

        if self.is_root():
            if authenticate:
                self._authenticate()

            message = "*** This is: '" + self.program + "'"
            if self.identifier and self.identifier != self.program:
                message += ", id: '" + self.identifier + "'"
            message += ", version: '" + self.version + "'"
            self.write(message)
            self.write('    Date: ' + str(datetime.now().date()) +
                       ' ' + str(datetime.now().time())[:8])
            self.write('    Path: ' + "'" + str(self.path) + "'")
            self.write('=== Pre-processing')
            self._exe_time_start = time()
            
        return ok

    def epilog(self) -> bool:
        ok = True
        for node in self.followers:
            if node:
                if not node.epilog():
                    ok = False

        if self.is_root():
            message = "'" + self.program + "' is successfully completed\n"
            exe_time = time() - self._exe_time_start
            if exe_time >= self._min_exe_time_shown:
                self.write('    Execution time: ' + format(round(exe_time, 2)))
            self.write('*** ' + message)

        if logger.handlers:
            logger.info('')
            logger.handlers = []
        sys.stdout.flush()
    
        return ok

    def load(self) -> bool:
        ok = True
        # import json
        # f = os.path.join((self.path,'data.json')
        # json.load(self.data, open(f, 'r'))
        return ok

    def save(self) -> bool:
        ok = True
        # import json
        # f = os.path.join(self.path, self.identifier + '.data.json')
        # json.dump(self.data, open(f, 'w'))
        return ok

    def initial_condition(self) -> bool:
        ok = True
        # super().initial_condition()        # use it in derived classes
        return ok

    def update_nonlinear(self) -> bool:
        ok = True
        # super().update_nonlinear()         # use it in derived classes
        return ok

    def update_transient(self) -> bool:
        ok = True
        # ok = super().update_transient()    # use it in derived classes
        return ok

    def pre(self, **kwargs: Any) -> bool:
        """
        Kwargs:
            Keyword arguments to be passed to pre() of followers

        Returns:
            False if data loading failed
        """
        ok = True
        for node in self.followers:
            if node:
                node.pre(**kwargs)
                if self.is_cooperator(node):
                    self.write(self.indent() + "    ['" + node.identifier +
                               "' is cooperator]")
        if self.root().followers:
            self.write('--- Pre (' + self.identifier + ')')

        if self.data is None:
            ok = self.load()
        self._pre_done = True
        sys.stdout.flush()
        
        return ok

    def task(self, **kwargs: Any) -> float:
        """
        Kwargs:
            Keyword arguments to be passed to task () of followers

        Returns:
            Residuum from range [0., 1.], indicating error of task
            OR
            Tuple of x array and y array in classes derived from Base
        """
        for node in self.followers:
            if node:
                node.task(**kwargs)
                if self.is_cooperator(node):
                    self.write(self.indent() + "    ['" + node.identifier +
                               "' is cooperator]")
        if self.root().followers:
            self.write('--- Task (' + self.identifier + ')')
        self._task_done = True
        sys.stdout.flush()
        
        return 0.0

    def post(self, **kwargs: Any) -> bool:
        """
        Kwargs:
            Keyword arguments to be passed to post() of followers

        Returns:
            False if data saving failed
        """
        ok = True
        for node in self.followers:
            if node:
                node.post(**kwargs)
                if self.is_cooperator(node):
                    self.write(self.indent() + "    ['" + node.identifier +
                               "' is cooperator]")
        if self.root().followers:
            self.write('--- Post (' + self.identifier + ')')
        if self.data is None:
            ok = self.save()
        self._post_done = True
        sys.stdout.flush()
        
        return ok

    def control(self, **kwargs: Any) \
            -> Union[float,              # residuum of children of Base or Loop
                     Dict[str, Any],    # metrics of train of BoxModel children
                     Float2D,            # y of prediction of BoxModel children
                     Tuple[Float1D, Float1D], # x_opt+y_opt of Min., Max., Inv. 
                     Tuple[Float1D, Float2D],     # x_ref, dy/dx of Sensitivity 
                     Tuple[Float2D, Float2D]]:             # x and y of Forward 
                                                                      
        """
        Kwargs:
            Keyword arguments to be passed to task() of this object

        Returns:
            Residuum from range [0., 1.], indicating error of task
                Base or Loop and its children
            OR 
            metrics of training for all children of BoxModel 
                exclusive White if x is None,
                see fifth code line of BoxModel.task()
            OR 
            y array of prediction of BoxModel and its children
            OR            
            x and y arrays of prediction of Forward and its children
        """
        if self.is_root():
            exe_time = time() - self._exe_time_start
            if exe_time >= self._min_exe_time_shown:
                self.write('    Execution time: {:2f} s'.format(round(exe_time,
                                                                      2)))
            self._exe_time_start = time()

        if self.is_root():
            self.write('=== Task-processing')
        task_result: Union[float,                        
                           Dict[str, Any],         
                           Float2D,
                           Tuple[Float1D, Float1D],
                           Tuple[Float1D, Float2D],
                           Tuple[Float2D, Float2D]] = self.task(**kwargs)
                                                              
        if self.is_root():
            exe_time = time() - self._exe_time_start
            if exe_time >= self._min_exe_time_shown:
                self.write('    Execution time: {:2f} s'.format(round(exe_time,
                                                                      2)))
            self._exe_time_start = time()
        self.write('=== Post-processing')
        
        return task_result
