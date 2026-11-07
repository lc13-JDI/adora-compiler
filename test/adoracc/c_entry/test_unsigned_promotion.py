"""AST-directed, type-preserving workaround for typedef-sensitive cgeist casts."""
import importlib.util
from pathlib import Path
import sys

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('c_entry', root/'tools/adoracc/c_entry.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def ast(spelling='uint16_t', underlying='unsigned short', kind='ImplicitCastExpr'):
    return {'kind':kind, 'castKind':'IntegralCast', 'type':{'qualType':'int'},
            'inner':[{'kind':'DeclRefExpr', 'type':{'qualType':spelling, 'desugaredQualType':underlying},
                      'range':{'begin':{'offset':1,'tokLen':1},'end':{'offset':1,'tokLen':1}}}]}

text, edits = m.normalize_unsigned_promotions(ast(), '(v)')
assert text == '(((unsigned short)(v)))', text
assert edits == [dict(begin=1,end=2,from_type='uint16_t',to_type='int',value_type='unsigned short')]
assert m.normalize_unsigned_promotions(ast('unsigned short'), '(v)') == ('(v)', [])
assert m.normalize_unsigned_promotions(ast('int16_t','short'), '(v)') == ('(v)', [])
assert m.normalize_unsigned_promotions(ast(kind='CStyleCastExpr'), '(v)')[0] == text
bad = ast(); bad['inner'][0]['range']['begin'] = {'spellingLoc':{'offset':1}}
try:
    m.normalize_unsigned_promotions(bad, '(v)')
except ValueError:
    pass
else:
    raise AssertionError('ambiguous source range accepted')
proof = dict(schema='adora.guarded-inplace-large.v1', value_mode='range', value_nonzero=False,
             range_lo=32768, range_hi=60000)
m.validate_large_value_proof(proof)
for field in ('value_mode', 'value_nonzero', 'range_lo', 'range_hi'):
    bad = dict(proof); del bad[field]
    try:
        m.validate_large_value_proof(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('missing proof field accepted: ' + field)
for changes in (dict(value_mode='direct'),dict(value_nonzero=True),dict(range_lo=0),dict(range_hi=65536)):
    try:
        m.validate_large_value_proof(dict(proof, **changes))
    except ValueError:
        pass
    else:
        raise AssertionError('contradictory value proof accepted')
print('unsigned promotion normalization tests passed')
