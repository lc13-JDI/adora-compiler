// RUN: rm -rf %t && mkdir -p %t
// RUN: python3 %S/test_guarded_stream.py --root %S/../../.. --tools $(dirname %cgra-mapper) --work %t
// RUN: python3 %S/test_remap_adapter.py
module {}
