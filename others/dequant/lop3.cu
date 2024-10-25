template<>structFastInterleavedAndBiasedNumericArrayConverter<half_t, uint4b_t, 8> {
using result_type= Array<half_t, 8>;
using source_type= Array<uint4b_t, 8>;

    CUTLASS_DEVICE
static result_typeconvert(source_typeconst& source)
    {
        result_type result;

        uint32_t*      h=reinterpret_cast<uint32_t*>(&result);
        // i4s = {e7,e5,e3,e1,e6,e4,e2,e0}
        // 这里和cutlass Array数据结构的实现相关，Array<uint4b_t, 8>实际上只有
        // 一个private成员变量Storage storage[kStorageElements]，代表一块连续的内存。其他都是static const
        // 成员，并且在编译器实现求值；因此source引用或指针，指向的实际就是storage；对于Array<uint4b_t, 8>来说，
        // storage是uint32_t；
        uint32_tconst i4s=reinterpret_cast<uint32_tconst&>(source);

        // First, we extract the i4s and construct an intermediate fp16 number.
        staticconstexpruint32_t immLut= (0xf0& 0xcc)| 0xaa;// 0b11101010
        staticconstexpruint32_t BOTTOM_MASK= 0x000f000f;// 0xf -> 0b1111 select 0,4
        staticconstexpruint32_t TOP_MASK= 0x00f000f0;// select 1,5
        staticconstexpruint32_t I4s_TO_F16s_MAGIC_NUM= 0x64006400;// 1024
        // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
        // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
        // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
        // elt_67 to fp16 without having to shift them to the bottom bits before hand.
        // NOTE: uint4b_t keep 4 bits in low 4bits of uint8_t's 8 bits, the internal storage is 8bits uint8_t.
        // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
        // immediately before required.
        // 首先右移8位，获得top_i4s，这个用来在不改变mask的情况下，获取e7~e4
        // {e7,e5,e3,e1,e6,e4,e2,e0} -> shift 8 -> {0x0,0x0,e7,e5,e3,e1,e6,e4}
        constuint32_t top_i4s = i4s >> 8;
        // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
        asmvolatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[0])
        : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
        // NOTE: 0x64[e3]064[e2]0 需要注意的是这时e3和e2是被保存在各自两个低字节的【高4bits】的
        // 这也是后续为什么要使用fma指令来还原原值的原因！注意，保存在高4bits，事实就是y*16（2^4=16）
        asmvolatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[1])
        : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
        asmvolatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[2])
        : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
        // NOTE: 0x64[e7]064[e6]0 需要注意的是这时e7和e6是被保存在各自两个低字节的【高4bits】的
        // 这也是后续为什么要使用fma指令来还原原值的原因！注意，保存在高4bits，事实就是y*16（2^4=16）
        asmvolatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[3])
        : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

        // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
        // half2 ctor. In this case, I chose performance reliability over code readability.
        // This is the half2 {1032, 1032} represented as an integer.
        staticconstexpruint32_t FP16_TOP_MAGIC_NUM= 0x64086408;
        // This is the half2 {1 / 16, 1 / 16} represented as an integer.
        staticconstexpruint32_t ONE_SIXTEENTH= 0x2c002c00;
        // This is the half2 {-72, -72} represented as an integer.
        // 个人理解: -72 = -64 - 8, massita expr 1024/16 + ((x+8)*16)/16 - 64 - 8 = x
        // Y_FP16 = 1024 + (x+8)*16, x = Y_FP16/16 - 64 - 8
        // (1024 + (x+8)*16)/16 = 64 + x + 8
        staticconstexpruint32_t NEG_72= 0xd480d480;

        // Finally, we construct the output numbers.
        // NOTE: uint4b_t keep 4 bits in low 4bits of uint8_t's 8 bits, the internal storage is 8bits uint8_t.
        // Convert elt_01
        asmvolatile("sub.f16x2 %0, %1, %2;\n": "=r"(h[0]): "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
        // Convert elt_23
        asmvolatile("fma.rn.f16x2 %0, %1, %2, %3;\n": "=r"(h[1]): "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
        // Convert elt_45
        asmvolatile("sub.f16x2 %0, %1, %2;\n": "=r"(h[2]): "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
        // Convert elt_67
        asmvolatile("fma.rn.f16x2 %0, %1, %2, %3;\n": "=r"(h[3]): "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));

        return result;
    }
};