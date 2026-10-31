This is Python version 3.16.0 alpha 1+dev (Experimental No-GIL + JIT + ``@strict``/``@struct``)
===============================================================================================

.. image:: https://github.com/python/cpython/actions/workflows/build.yml/badge.svg?branch=main&event=push
   :alt: CPython build status on GitHub Actions
   :target: https://github.com/python/cpython/actions

.. image:: https://dev.azure.com/python/cpython/_apis/build/status/Azure%20Pipelines%20CI?branchName=main
   :alt: CPython build status on Azure DevOps
   :target: https://dev.azure.com/python/cpython/_build/latest?definitionId=4&branchName=main

.. image:: https://img.shields.io/badge/discourse-join_chat-brightgreen.svg
   :alt: Python Discourse chat
   :target: https://discuss.python.org/


Experimental Free-Threaded Tier 2 / JIT & ``@strict`` / ``@struct`` Runtime
---------------------------------------------------------------------------

This branch (``experimental-nogil-jit``) combines **PEP 703 Free-Threading** (``--disable-gil``),
an enhanced **Tier 2 Micro-Op Abstract Interpreter**, a **Native AArch64 Copy-and-Patch JIT**,
an **Opt-In Sound Gradual Typing & Sealed Slot System** (`Lib/strict.py <Lib/strict.py>`_: ``@strict``, ``@struct``,
``strict_mode``, ``strict_module``), and a **C-Level Work-Stealing Auto-Parallel Dispatch Engine**.

* **Full Architecture, Optimizer Passes, Special Cases & Roadmap:**
  See `InternalDocs/nogil_jit_architecture.md <InternalDocs/nogil_jit_architecture.md>`_.
* **Opt-In Sound Gradual Typing Module:**
  See `Lib/strict.py <Lib/strict.py>`_.
* **Paired Benchmark & Boundary Enforcement Suite:**
  See `Tools/benchmarks/bench_strict_duplication.py <Tools/benchmarks/bench_strict_duplication.py>`_.

Key Architectural Highlights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Zero-Indirection Free-Threaded Runtime (** ``--disable-gil`` **):**
   Zero-indirection main-thread Thread-Local Bytecode (TLBC), worker TLBC specialization inheritance,
   unshared-local container fast paths (bypassing critical sections when ``ob_ref_shared == 0``),
   and a unified 1- and 2-digit (``<= 60``-bit) ``PyLong`` per-thread freelist.
2. **Partial Escape Analysis / Scalar Replacement (** ``eliminate_non_escaping_struct_allocs`` **):**
   Eliminates non-escaping ``@struct`` heap allocations inside hot loops by inlining ``__init__``,
   forwarding field writes directly to downstream ``_LOAD_ATTR_SLOT`` consumers on the abstract stack,
   and removing ``_ALLOCATE_OBJECT`` / ``_CREATE_INIT_FRAME`` / ``_MAKE_HEAP_SAFE`` (**5.01x** faster
   on ``dataclass_inst``, **25.74x** faster than Tier 1).
3. **Perceus-Style Reference-Unique In-Place Reuse & FMA Fusion:**
   Exploits PEP 703 biased reference counting (``_Py_IsOwnedByCurrentThread(op) && ob_ref_local == 1 && ob_ref_shared == 0``)
   to mutate ``PyFloatObject`` and 60-bit compact ``PyLongObject`` boxes in place across arithmetic chains,
   slot updates (``_STORE_ATTR_SLOT_FLOAT_INPLACE``, ``_LOAD_ATTR_SLOT_NO_POP``), hardware Fused Multiply-Add
   (``_BINARY_OP_FMA_FLOAT_INPLACE``), 60-bit compact-int bitwise/divmod (``_BINARY_OP_BITWISE_INT``),
   and mixed float-int arithmetic (``_BINARY_OP_ARITH_FLOAT_INT``).
4. **Basic Block Versioning via Loop Pre-Header Peeling (** ``hoist_loop_preheader_guards`` **):**
   Peels up to 64 micro-ops of the first loop iteration into a trace pre-header to prove steady-state local
   types and non-nullness (``_LOAD_FAST`` -> ``_LOAD_FAST_BORROW``) and hoist invariant type/container guards.
5. **Opt-In Sound Gradual Typing (** ``@strict`` **,** ``@struct`` **,** ``strict_mode`` **,** ``strict_module`` **):**
   Enforces primitive and slot type annotations in C via ``_Py_STRICT_SLOT_*`` flags on ``PyMemberDef``
   and 64-bit ``_co_strict_sig`` bitmasks on ``PyCodeObject`` (``_CHECK_FUNCTION_STRICT_ARGS``), sealing
   classes with ``Py_TPFLAGS_IMMUTABLETYPE`` and seeding Tier 2 with guard-free static types while catching
   type violations across third-party boundaries via ``sys.monitoring``.

Quickstart: Building & Benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Configure and build free-threaded CPython with the native Copy-and-Patch JIT
    mkdir -p build-nogil && cd build-nogil
    ../configure --disable-gil --enable-experimental-jit
    make -j10

    # Run the Tier 2 / JIT optimizer regression suite (335 tests)
    ./python.exe -m test test_capi -m test_opt

    # Run the 11 paired workloads + 8 negative boundary enforcement tests
    ./python.exe ../Tools/benchmarks/bench_strict_duplication.py

Benchmark Summary (Apple Silicon AArch64, ``--disable-gil --enable-experimental-jit``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

========================  =======================  ===============  =========================  ==================  ======================
Workload                  Tier 1 (``PYTHON_JIT=0``)  Undecorated JIT  ``@strict`` / ``@struct``  Speedup vs. Tier 1  Speedup vs. Undec. JIT
========================  =======================  ===============  =========================  ==================  ======================
``dataclass_inst``        123.56 ms                26.75 ms         **4.80 ms**                **25.74x**          **5.57x**
``object_attr_churn``     135.20 ms                48.52 ms         **10.17 ms**               **13.29x**          **4.77x**
``parallel_4threads``     248.10 ms                79.95 ms         **19.67 ms**               **12.61x**          **4.06x**
``nbody_float``           376.82 ms                185.01 ms        **40.24 ms**               **9.36x**           **4.60x**
``attr_loop``             74.80 ms                 35.87 ms         **11.23 ms**               **6.66x**           **3.19x**
``fib_iter``              59.42 ms                 22.97 ms         **15.46 ms**               **3.84x**           **1.49x**
``call_overhead``         110.27 ms                49.70 ms         **33.17 ms**               **3.32x**           **1.50x**
``mixed_dispatch``        94.77 ms                 34.23 ms         **33.03 ms**               **2.87x**           **1.04x**
``arith_loop``            126.44 ms                78.59 ms         **51.40 ms**               **2.46x**           **1.53x**
``list_comp``             111.79 ms                59.17 ms         **57.40 ms**               **1.95x**           **1.03x**
**Geometric Mean (11)**   **122.90 ms**            **56.71 ms**     **30.22 ms**               **4.06x**           **1.88x**
========================  =======================  ===============  =========================  ==================  ======================

Selected Literature References
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Copy-and-Patch JIT:** H. Xu & F. Kjolstad, *"Copy-and-Patch Compilation: A fast compilation algorithm for high-level languages and bytecode,"* OOPSLA 2021 (`doi:10.1145/3485513 <https://doi.org/10.1145/3485513>`_).
* **Partial Escape Analysis:** L. Stadler, T. Würthinger, & H. Mössenböck, *"Partial Escape Analysis and Scalar Replacement for Java,"* CGO 2014 (`doi:10.1145/2544137.2544157 <https://doi.org/10.1145/2544137.2544157>`_).
* **Tracing Allocation Removal:** C. F. Bolz et al., *"Allocation Removal by Partial Evaluation in a Tracing JIT,"* PEPM 2011 (`doi:10.1145/1929501.1929508 <https://doi.org/10.1145/1929501.1929508>`_).
* **Basic Block Versioning:** M. Chevalier-Boisvert & M. Feeley, *"Simple and Effective Type Check Removal through Lazy Basic Block Versioning,"* ECOOP 2015 (`doi:10.4230/LIPIcs.ECOOP.2015.101 <https://doi.org/10.4230/LIPIcs.ECOOP.2015.101>`_).
* **Reference-Unique In-Place Reuse:** A. Reinking, N. Xie, L. de Moura, & D. Leijen, *"Perceus: Garbage Free Reference Counting with Reuse,"* PLDI 2021 (`doi:10.1145/3453483.3454032 <https://doi.org/10.1145/3453483.3454032>`_).
* **Biased Reference Counting & Free-Threading:** J. Choi, T. Shull, & J. Torrellas, *"Biased Reference Counting: Minimizing Atomic Operations in Garbage Collection,"* PACT 2018 (`doi:10.1145/3243176.3243195 <https://doi.org/10.1145/3243176.3243195>`_); S. Gross, `PEP 703 <https://peps.python.org/pep-0703/>`_.
* **Sound Gradual Typing & Static Python:** A. Rastogi et al., *"Safe & Efficient Gradual Typing for TypeScript,"* POPL 2015 (`doi:10.1145/2676726.2676971 <https://doi.org/10.1145/2676726.2676971>`_); M. M. Vitousek et al., *"Big Types in Little Runtime,"* POPL 2017 (`doi:10.1145/3009837.3009849 <https://doi.org/10.1145/3009837.3009849>`_); Meta Engineering, `Cinder Static Python <https://github.com/facebookincubator/cinder>`_.

----

Copyright © 2001 Python Software Foundation.  All rights reserved.

See the end of this file for further copyright and license information.

.. contents::

General Information
-------------------

- Website: https://www.python.org
- Source code: https://github.com/python/cpython
- Issue tracker: https://github.com/python/cpython/issues
- Documentation: https://docs.python.org
- Developer's Guide: https://devguide.python.org/

Contributing to CPython
-----------------------

For more complete instructions on contributing to CPython development,
see the `Developer Guide`_.

.. _Developer Guide: https://devguide.python.org/

Using Python
------------

Installable Python kits, and information about using Python, are available at
`python.org`_.

.. _python.org: https://www.python.org/

Build Instructions
------------------

On Unix, Linux, BSD, macOS, and Cygwin::

    ./configure
    make
    make test
    sudo make install

This will install Python as ``python3``.

You can pass many options to the configure script; run ``./configure --help``
to find out more.  On macOS case-insensitive file systems and on Cygwin,
the executable is called ``python.exe``; elsewhere it's just ``python``.

Building a complete Python installation requires the use of various
additional third-party libraries, depending on your build platform and
configure options.  Not all standard library modules are buildable or
usable on all platforms.  Refer to the
`Install dependencies <https://devguide.python.org/getting-started/setup-building.html#build-dependencies>`_
section of the `Developer Guide`_ for current detailed information on
dependencies for various Linux distributions and macOS.

On macOS, there are additional configure and build options related
to macOS framework and universal builds.  Refer to `Mac/README.rst
<https://github.com/python/cpython/blob/main/Mac/README.rst>`_.

On Windows, see `PCbuild/readme.txt
<https://github.com/python/cpython/blob/main/PCbuild/readme.txt>`_.

To build Windows packages, see `PC/layout/README.md
<https://github.com/python/cpython/blob/main/PC/layout/README.md>`_.

If you wish, you can create a subdirectory and invoke configure from there.
For example::

    mkdir debug
    cd debug
    ../configure --with-pydebug
    make
    make test

(This will fail if you *also* built at the top-level directory.  You should do
a ``make clean`` at the top-level first.)

To get an optimized build of Python, run ``configure --enable-optimizations``
before you run ``make``.  This sets the default make targets up to enable
Profile Guided Optimization (PGO) and may be used to auto-enable Link Time
Optimization (LTO) on some platforms.  For more details, see the sections
below.

Profile Guided Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

PGO takes advantage of recent versions of the GCC or Clang compilers.  If used,
either via ``configure --enable-optimizations`` or by manually running
``make profile-opt`` regardless of configure flags, the optimized build
process will perform the following steps:

The entire Python directory is cleaned of temporary files that may have
resulted from a previous compilation.

An instrumented version of the interpreter is built, using suitable compiler
flags for each flavor. Note that this is just an intermediary step.  The
binary resulting from this step is not good for real-life workloads as it has
profiling instructions embedded inside.

After the instrumented interpreter is built, the Makefile will run a training
workload.  This is necessary in order to profile the interpreter's execution.
Note also that any output, both stdout and stderr, that may appear at this step
is suppressed.

The final step is to build the actual interpreter, using the information
collected from the instrumented one.  The end result will be a Python binary
that is optimized; suitable for distribution or production installation.


Link Time Optimization
^^^^^^^^^^^^^^^^^^^^^^

Enabled via configure's ``--with-lto`` flag.  LTO takes advantage of the
ability of recent compiler toolchains to optimize across the otherwise
arbitrary ``.o`` file boundary when building final executables or shared
libraries for additional performance gains.


What's New
----------

We have a comprehensive overview of the changes in the `What's new in Python
3.16 <https://docs.python.org/3.16/whatsnew/3.16.html>`_ document.  For a more
detailed change log, read `Misc/NEWS
<https://github.com/python/cpython/tree/main/Misc/NEWS.d>`_, but a full
accounting of changes can only be gleaned from the `commit history
<https://github.com/python/cpython/commits/main>`_.

If you want to install multiple versions of Python, see the section below
entitled "Installing multiple versions".


Documentation
-------------

`Documentation for Python 3.16 <https://docs.python.org/3.16/>`_ is online,
updated daily.

It can also be downloaded in many formats for faster access.  The documentation
is downloadable in HTML, EPUB, and reStructuredText formats; the latter version
is primarily for documentation authors, translators, and people with special
formatting requirements.

For information about building Python's documentation, refer to `Doc/README.rst
<https://github.com/python/cpython/blob/main/Doc/README.rst>`_.


Testing
-------

To test the interpreter, type ``make test`` in the top-level directory.  The
test set produces some output.  You can generally ignore the messages about
skipped tests due to optional features which can't be imported.  If a message
is printed about a failed test or a traceback or core dump is produced,
something is wrong.

By default, tests are prevented from overusing resources like disk space and
memory.  To enable these tests, run ``make buildbottest``.

If any tests fail, you can re-run the failing test(s) in verbose mode.  For
example, if ``test_os`` and ``test_gdb`` failed, you can run::

    make test TESTOPTS="-v test_os test_gdb"

If the failure persists and appears to be a problem with Python rather than
your environment, you can `file a bug report
<https://github.com/python/cpython/issues>`_ and include relevant output from
that command to show the issue.

See `Running & Writing Tests <https://devguide.python.org/testing/run-write-tests.html>`_
for more on running tests.

Installing multiple versions
----------------------------

On Unix and Mac systems if you intend to install multiple versions of Python
using the same installation prefix (``--prefix`` argument to the configure
script) you must take care that your primary python executable is not
overwritten by the installation of a different version.  All files and
directories installed using ``make altinstall`` contain the major and minor
version and can thus live side-by-side.  ``make install`` also creates
``${prefix}/bin/python3`` which refers to ``${prefix}/bin/python3.X``.  If you
intend to install multiple versions using the same prefix you must decide which
version (if any) is your "primary" version.  Install that version using
``make install``.  Install all other versions using ``make altinstall``.

For example, if you want to install Python 2.7, 3.6, and 3.15 with 3.15 being the
primary version, you would execute ``make install`` in your 3.15 build directory
and ``make altinstall`` in the others.


Release Schedule
----------------

See `PEP 826 <https://peps.python.org/pep-0826/>`__ for Python 3.16 release details.


Copyright and License Information
---------------------------------


Copyright © 2001 Python Software Foundation.  All rights reserved.

Copyright © 2000 BeOpen.com.  All rights reserved.

Copyright © 1995-2001 Corporation for National Research Initiatives.  All
rights reserved.

Copyright © 1991-1995 Stichting Mathematisch Centrum.  All rights reserved.

See the `LICENSE <https://github.com/python/cpython/blob/main/LICENSE>`_ for
information on the history of this software, terms & conditions for usage, and a
DISCLAIMER OF ALL WARRANTIES.

This Python distribution contains *no* GNU General Public License (GPL) code,
so it may be used in proprietary projects.  There are interfaces to some GNU
code but these are entirely optional.

All trademarks referenced herein are property of their respective holders.
