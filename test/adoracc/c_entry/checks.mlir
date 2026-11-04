// RUN: python3 %S/test_c_entry.py %S/../../..
// RUN: python3 -O %S/test_c_entry.py %S/../../..
// Host checks must remain active under Python optimization.
module {}
