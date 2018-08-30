# OOPT-Raman

oopt-raman is a python library for evaluating the stimulated Raman scattering in optical communication systems.
The Raman solver is implemented solving the set of pump and probe ordirary differential equations.

### Installation

`oopt-raman` can be installed from the repository.
```sh
$ git clone https://github.com/Telecominfraproject/oopt-raman.git
$ cd oopt-raman
$ git checkout develop
$ python3 setup.py install
```

### Instructions on how run the example

The example needs `matplotlib`. To install it, you can use pip.
```sh
$ pip install matplotlib
```

Then you have to navigate to the example folder.
```sh
$ cd /oopt-raman/examples
```

Launch the example
```sh
$ python3  main_c_band_raman_amplifier.py
```
