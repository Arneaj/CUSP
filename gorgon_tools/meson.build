# look inside these directories for more meson.build files 
subdir('geomagnetic')
subdir('magnetosphere')

# include the following direcotries
install_subdir('ionosphere', install_dir: py.get_install_dir() / 'gorgon_tools')
install_subdir('legacy', install_dir: py.get_install_dir() / 'gorgon_tools')
install_subdir('misc', install_dir: py.get_install_dir() / 'gorgon_tools')
install_subdir('models', install_dir: py.get_install_dir() / 'gorgon_tools')
install_subdir('scripts', install_dir: py.get_install_dir() / 'gorgon_tools')
install_subdir('solar_wind', install_dir: py.get_install_dir() / 'gorgon_tools')
install_subdir('tests', install_dir: py.get_install_dir() / 'gorgon_tools')
install_subdir('geomagnetic/data', install_dir: py.get_install_dir() / 'gorgon_tools/geomagnetic/')

python_sources = [
    '__init__.py',
    'version.py'
]

py.install_sources(
  python_sources,
  subdir: 'gorgon_tools'
)
