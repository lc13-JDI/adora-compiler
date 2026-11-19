// Read-only observer for the byte-locked 128-bit CSTORE target.
#include <vpi_user.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
using Signal = vpiHandle;
struct Record {
    uint32_t cycle;
    uint8_t kind, flags;
    uint16_t aux;
    uint32_t arg;
    uint32_t data[4];
};
static_assert(sizeof(Record) == 28, "trace ABI");
struct Channel {
    Signal valid{}, ready{}, address{}, length{}, size{}, burst{}, id{};
    Signal data{}, strb{}, last{}, resp{};
};
struct Bank { Signal en, we, address, din, dout; };
FILE *out;
std::string missing;
std::array<Channel, 10> channels;
std::vector<Bank> banks;
std::vector<Signal> starts, dones;
Signal rst, config_done;
int last_reset = -1, last_config = -1, last_dones = -1;
int output_bank = -1;
bool detailed = false;
Signal signal(const std::string &name) {
    auto result = vpi_handle_by_name(const_cast<char *>(name.c_str()), nullptr);
    if (!result && missing.empty()) missing = name;
    return result;
}
uint32_t read(Signal s) {
    if (!s) return 0;
    s_vpi_value v{};
    v.format = vpiIntVal;
    vpi_get_value(s, &v);
    return v.value.integer;
}
void emit(uint32_t cycle, int kind, int flags=0, int aux=0, uint32_t arg=0,
          uint32_t d0=0, uint32_t d1=0, uint32_t d2=0, uint32_t d3=0) {
    const Record r{cycle, uint8_t(kind), uint8_t(flags), uint16_t(aux), arg, {d0,d1,d2,d3}};
    std::fwrite(&r,sizeof(r),1,out);
}
}
extern "C" const char *canvas_open(const char *path, const char *top, int bank_count, int tiles) {
    const uint16_t endian=1;
    if (*reinterpret_cast<const uint8_t*>(&endian)!=1) return "little-endian host required";
    out = std::fopen(path,"wb");
    if (!out) return "cannot open wire trace";
    const std::string root = std::string(top)+".";
    rst = signal(root+"rst");
    config_done = signal(root+"uut.cgra.cfgCtrl_io_done");
    const char *names[]={"aw","w","b","ar","r"};
    for (int i=0;i<10;++i) {
        bool axi=i<5;
        int channel=i%5;
        std::string p=root+(axi?"axi_":"axil_")+names[channel];
        auto &c=channels[i];
        c.valid=signal(p+"valid"); c.ready=signal(p+"ready");
        if(channel==0 || channel==3) {
            c.address=signal(p+"addr");
            if(axi) { c.length=signal(p+"len"); c.size=signal(p+"size");
                      c.burst=signal(p+"burst"); c.id=signal(p+"id"); }
        } else if(channel==1 || channel==4) {
            c.data=signal(p+"data");
            if(channel==1) c.strb=signal(p+"strb"); else c.resp=signal(p+"resp");
            if(axi) c.last=signal(p+"last");
            if(axi && channel==4) c.id=signal(p+"id");
        } else {
            c.resp=signal(p+"resp");
            if(axi) c.id=signal(p+"id");
        }
    }
    for(int b=0;b<bank_count;++b) {
        auto p=root+"uut.cgra_io_srams_iob_"+std::to_string(b)+"_";
        banks.push_back({signal(p+"en"),signal(p+"we"),signal(p+"addr"),signal(p+"din"),signal(p+"dout")});
    }
    for(int t=0;t<tiles;++t) {
        const auto p=root+"uut.cgra.cgra_io_";
        starts.push_back(signal(p+"start_"+std::to_string(t)));
        dones.push_back(signal(p+"done_"+std::to_string(t)));
    }
    return missing.empty()?nullptr:missing.c_str();
}
extern "C" void canvas_call(uint32_t cycle, int begin, int frame, int block, int bank, int sample) {
    if (begin) { output_bank=bank; detailed=sample; }
    emit(cycle,begin?16:17,sample,frame,block,bank);
    if (!begin) { output_bank=-1; detailed=false; std::fflush(out); }
}
extern "C" void canvas_wait(uint32_t cycle, int mask, int status, int response_cycle) {
    emit(cycle,18,0,mask,status,response_cycle);
}
extern "C" void canvas_tick(uint32_t cycle) {
    int reset=read(rst);
    if(reset!=last_reset) { emit(cycle,10,reset); last_reset=reset; }
    if(reset) return;
    for(int i=0;i<10;++i) {
        const auto &c=channels[i];
        if(!read(c.valid) || !read(c.ready)) continue;
        uint32_t data[4]{};
        if(c.data) {
            s_vpi_value v{};
            v.format=vpiVectorVal;
            vpi_get_value(c.data,&v);
            for(int w=0;w<(i<5?4:1);++w) data[w]=v.value.vector[w].aval;
        }
        if(c.address) {
            emit(cycle,i,read(c.burst),read(c.length),read(c.address),read(c.size),read(c.id));
        } else {
            emit(cycle,i,read(c.last),read(c.strb),read(c.resp)|(read(c.id)<<8),
                 data[0],data[1],data[2],data[3]);
        }
    }
    int cfg=read(config_done), sm=0, dm=0;
    if(cfg!=last_config) { emit(cycle,11,cfg); last_config=cfg; }
    for(size_t t=0;t<starts.size();++t) { sm|=read(starts[t])<<t; dm|=read(dones[t])<<t; }
    if(sm || dm!=last_dones) { emit(cycle,12,0,sm,dm); last_dones=dm; }
    if(output_bank<0) return;
    for(int b=detailed?0:output_bank;b<(detailed?int(banks.size()):output_bank+1);++b) {
        const auto &s=banks[b];
        if(read(s.en)) emit(cycle,13,read(s.we),b,read(s.address),read(s.din),read(s.dout));
    }
}
extern "C" void canvas_close() { if(out) { std::fclose(out); out=nullptr; } }
