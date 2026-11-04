import os,ast,collections,csv,importlib.util,json,re,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=Path(os.environ.get("ADORA_ROOT",D.parents[2]))
spec=importlib.util.spec_from_file_location('alignment',R/'test/cgra-mapper/cstore_contract/test_cstore_alignment.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
def graph(p):
 nodes={};ins={}
 for line in p.read_text().splitlines():
  m=re.match(r'(\w+)\[opcode\s*=',line)
  if m:nodes[m[1]]=dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"',line))
  m=re.match(r'(\w+) -> (\w+)\[.*operand\s*=\s*(-?\d+)',line)
  if m and int(m[3])>=0:ins.setdefault(m[2],{})[int(m[3])]=m[1]
 return nodes,ins
def inspect(run):
 meta=json.loads((run/'command.json').read_text());assert meta['rc']==0
 kernel=meta['kernel'];mapping=run/(kernel+'_map_result');adg=json.loads((R/'test/spec/cgra_cstore_vitra/artifacts/adg.json').read_text());nodes,ins=graph(mapping/('before_map_'+kernel+'_CDFG.dot'))
 tree=ast.parse((run/'mapped.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name==kernel)
 seq={x:[] for x in ['idata','iptrs','odata','optrs']};mask=None
 for n in ast.walk(fn):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.attr=='append' and n.func.value.id in seq:
   target=n.func.value.id;v=n.args[0]
   if target in ['iptrs','optrs']:seq[target].append([ast.literal_eval(a) for a in v.args])
   else:
    assert isinstance(v,ast.Subscript);seq[target].append(dict(arg=v.value.id,slice=ast.unparse(v.slice)))
  if isinstance(n,ast.keyword) and n.arg=='iob_en':mask=ast.literal_eval(n.value)
 packets=dict(h.read_pytest_packets(run/'mapped.py',kernel,adg));routes=list(csv.DictReader((mapping/'mapped_routes.tsv').open(),delimiter='\t'));bindings=[];used=set()
 for line in (mapping/'mapped_dfgio.txt').read_text().splitlines():
  if line.startswith('#'):continue
  tag,name,physical,index,lat=[x.strip() for x in line.split(',')];physical,index,lat=int(physical),int(index),int(lat);op=tag.split('_')[0];m=re.match(r'(.*?)(\d+)$',name);n=m[1]+str(int(m[2])-1);attrs=nodes[n]
  instance,module=h.find_iob(adg,physical);fields={f:h.decode_controller_field(packets,instance,module,adg,f) for f in ['BaseAddr','UseAddr','UseEn','IsStore','Latency','II','Cycles0','Stride0']}
  assert fields['UseAddr']==int(op=='LOAD') and fields['UseEn']==0 and fields['IsStore']==int(op=='OUTPUT')
  stride,cycles=map(int,attrs['pattern'].split(','));assert fields['Cycles0']==cycles and fields['Stride0']==stride//2
  assert (stride,cycles)==((0,1) if op=='OUTPUT' else (0,meta['length']) if op=='LOAD' else (2,meta['length']))
  banks=adg['iob_to_spad_banks'][str(index)];bs=adg['iob_spad_bank_size'];offset=int(attrs['offset'].split(',')[1]);bytebase=fields['BaseAddr']*2-offset;bank=min(banks)+bytebase//bs;address=bank*bs
  assert bank in banks and bank not in used and bytebase%bs==0;used.add(bank)
  assert [address,int(attrs['size'])] in seq['optrs' if op=='OUTPUT' else 'iptrs']
  if op=='LOAD':
   rows=[r for r in routes if int(r['dst_adg'])==physical];assert len(rows)==1 and rows[0]['logical_operand']=='0';h.validate_route_configuration(packets,instance,module,adg,rows)
  assert all(int(r['arrival_latency'])==int(r['target_latency']) for r in routes)
  if op=='OUTPUT':
   output_routes=[r for r in routes if int(r['dst_adg'])==physical]
   assert len(output_routes)==1 and fields['Latency']==int(output_routes[0]['target_latency'])+(meta['length']-1)*fields['II']
  bindings.append(dict(node=n,opcode=op,iob=index,bank=bank,host_address=address,fields=fields,object=attrs['ref_name'],size=int(attrs['size']),element_offset=offset//2))
 assert set(i for i in range(48) if int.from_bytes(bytes(mask),'little')>>i&1)=={b['iob'] for b in bindings}
 assert len(seq['optrs'])==2 and sum(b['opcode']=='LOAD' for b in bindings)==2
 # Mapped graph includes actual replicated routing operators; count them separately.
 mappednodes=re.findall(r'^"([^"\n]+)"\[label', (mapping/'mapped_dfg.dot').read_text(),re.M)
 counts=collections.Counter(nodes[n]['opcode'].upper() for n in nodes)
 cfg=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('cfgbit_') for t in n.targets))
 accs=[]
 for name,a in nodes.items():
  if a['opcode'].upper()!='ACC':continue
  m=re.match(r'(.*?)(\d+)$',name);mapped=m[1]+str(int(m[2])+1)
  rows=[r for r in routes if r['dst_operation']=='ACC' and r['dst_dfg']==mapped]
  if not rows: rows=[r for r in routes if r['dst_operation']=='ACC' and int(r['dst_dfg'])==int(m[2])+1]
  assert len(rows)==1,(name,rows)
  physical=int(rows[0]['dst_adg']);inst=next(i for i in adg['instances'] if i['id']==physical)
  mod=next(m for m in adg['sub_modules'] if m['id']==inst['module_id'])['attributes']
  fields={f:h.decode_range(packets,inst,adg,*h.config_range(mod,cid,f)) for f,cid in mod['affine_ctrl_reg_cfg_id'].items()}
  assert fields['InitVal']==0 and fields['Cycles']==meta['length'] and fields['Repeats']==1 and fields['SkipFirst']==0
  assert fields['WI']==bindings[0]['fields']['II']
  accs.append(dict(node=name,physical=physical,fields=fields))
 assert len(accs)==2
 result=dict(**meta,accs=accs,bindings=bindings,transfers=seq,input_bytes=sum(p[1] for p in seq['iptrs']),output_bytes=sum(p[1] for p in seq['optrs']),iob_count=len(bindings),bank_count=len(used),logical_operations=dict(counts),mapped_nodes=len(mappednodes),mapped_compute_nodes=len(mappednodes)-len(bindings),route_count=len(routes),route_latency_sum=sum(int(r['route_latency']) for r in routes),config_packets=len(cfg)//3,config_bytes=len(cfg)*2)
 (run/'contract.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 for root in map(Path,sys.argv[1:]):
  for p in sorted(root.glob('*/command.json')):
   if json.loads(p.read_text())['rc']==0:
    r=inspect(p.parent);print(p.parent.name,r['iob_count'],r['input_bytes'],r['config_bytes'])
