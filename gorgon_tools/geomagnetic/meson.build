# include directory for numpy
incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy;  print(numpy.get_include())'],
  check : true
).stdout().strip()
inc_np = include_directories(incdir_numpy)

# include f2py
incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

py_dep = py.dependency()

# f2py fortran library
fortranobject_lib = static_library('_fortranobject',
  fortranobject_c,
  dependencies: py_dep,
  include_directories: [inc_np, inc_f2py],
)
fortranobject_dep = declare_dependency(
  link_with: fortranobject_lib,
  include_directories: [inc_np, inc_f2py],
)

numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

# combine fortran files
fortran_src = [
  'fortran_src/igrf_calc.f90'
]

# link and compile fortran code
fortran_module = custom_target('_fortran',
  input: fortran_src,
  output : ['_fortranmodule.c', '_fortran-f2pywrappers2.f90'],
  command : [py, '-m', 'numpy.f2py', '-m', '_fortran', '@INPUT@', '--build-dir', '@OUTDIR@']
)
py.extension_module('_fortran',
    fortran_src,
    fortran_module,
    c_args : numpy_nodepr_api,
    include_directories : inc_np,
    dependencies : fortranobject_dep,
    link_language : 'fortran',
    subdir : 'gorgon_tools' / 'geomagnetic',
    install : true
)

# python sources
python_sources = [
    '__init__.py',
    'benchmarks.py',
    'calcdeltaB.py',
    'CIM.py',
    'coordinates.py',
    'dB_import.py',
    'elecproject.py',
    'igrf.py',
    'visualise.py'
]

py.install_sources(
  python_sources,
  subdir: 'gorgon_tools' / 'geomagnetic'
)
