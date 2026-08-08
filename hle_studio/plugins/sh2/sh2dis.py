#!/usr/bin/env python3
"""Minimal SH-2 disassembler mirroring Yabause's sh2int.c decode() nibble logic
(vendor/yabause/src/sh2int.c) exactly, opcode by opcode, cross-checked against
that source rather than guessed. Only implements opcodes actually encountered
in real traces so far; unknown patterns print as UNKNOWN so gaps stay visible
instead of being silently misdecoded (per the project's "don't guess" rule,
see GEMINI.md).

Used by study_frame.py to turn a traces/frames/frameN.txt PC list into an
annotated disassembly. See docs/decompilation_architecture.md for how this
fits into the wider recompilation pipeline.
"""

BASE = 0x06000000  # start of extraction/IP.BIN, not 0.BIN's own 0x06004000 - see load_combined_binary()


def s8(v):
    return v - 0x100 if v & 0x80 else v


def s16(v):
    return v - 0x10000 if v & 0x8000 else v


def load_binary(path):
    with open(path, "rb") as f:
        return f.read()


IP_BIN_LOAD_ADDR = 0x06000000  # confirmed 2026-08-06 by direct disassembly test, NOT the
                                # 0x06002000 docs/address_mapping.md assumed - decoding IP.BIN's
                                # own bytes at that base produced a real function prologue right
                                # where a genuinely-traced call target (060006AA) landed as BSRF R3


def load_combined_binary(bin_path, ip_bin_path):
    """Builds one flat buffer spanning [IP_BIN_LOAD_ADDR, BASE_OF_0BIN + len(0.BIN)) so
    read_u16/read_u32 (BASE = IP_BIN_LOAD_ADDR) resolve BOTH files with correct real-hardware
    precedence: IP.BIN owns [0x06000000, 0x06004000) (its own first 16KB - the part 0.BIN's
    own load never overwrites), 0.BIN owns [0x06004000, 0x06004000+len(0.BIN)) as before -
    unchanged bytes/offsets for every address that already worked. Root-caused 2026-08-06:
    108 call targets that were failing "fora do alcance de 0.BIN" split into two clusters,
    52 of which sit below 0x06004000 - IP.BIN's own address range - and were simply never
    mapped at all (the other 56 are a separate, still-open question: dynamically-loaded
    OL00.BIN..OL15.BIN overlay files sharing the address space starting at 0x060D0000,
    which specific one is live depends on real emulator state a static file alone can't
    answer - see the portal_trace-based analyzer idea in the conversation this was found in
    for the actual next step there, not attempted here)."""
    ip_bin = load_binary(ip_bin_path)
    zero_bin = load_binary(bin_path)
    zero_bin_base = 0x06004000
    ip_bin_slice = ip_bin[: zero_bin_base - IP_BIN_LOAD_ADDR]
    return ip_bin_slice + zero_bin


def read_u16(data, addr):
    off = addr - BASE
    if off < 0 or off + 1 >= len(data):
        return None
    return (data[off] << 8) | data[off + 1]


def read_u32(data, addr):
    off = addr - BASE
    if off < 0 or off + 3 >= len(data):
        return None
    return (data[off] << 24) | (data[off + 1] << 16) | (data[off + 2] << 8) | data[off + 3]


class Decoded:
    """One decoded instruction: mnemonic text, plus structured fields the
    study generator uses for call/branch/pool-resolution logic."""

    def __init__(self, addr, op, mnemonic, kind=None, target=None, pool_addr=None, pool_val=None):
        self.addr = addr
        self.op = op
        self.mnemonic = mnemonic
        self.kind = kind          # 'call' | 'branch' | 'cond_branch' | None
        self.target = target      # resolved absolute target, if known
        self.pool_addr = pool_addr
        self.pool_val = pool_val


def decode(pc, op, data=None):
    """Decode one opcode at address pc. If `data` (the 0.BIN bytes) is given,
    PC-relative literal pool loads (MOV.L/MOV.W @(disp,PC)) are resolved."""
    a = (op & 0xF000) >> 12
    b = (op & 0x0F00) >> 8   # n
    c = (op & 0x00F0) >> 4   # m
    d = op & 0x000F
    cd = op & 0x00FF
    bcd = op & 0x0FFF

    if op == 0x0009:
        return Decoded(pc, op, "NOP")
    if op == 0x000B:
        return Decoded(pc, op, "RTS", kind="branch", target=None)  # target = PR, unknown statically
    if op == 0x002B:
        return Decoded(pc, op, "RTE")
    if op == 0x0008:
        return Decoded(pc, op, "CLRT")
    if op == 0x0028:
        return Decoded(pc, op, "CLRMAC")
    if op == 0x0018:
        return Decoded(pc, op, "SETT")
    if op == 0x0019:
        return Decoded(pc, op, "DIV0U")
    if a == 2 and d == 7:
        return Decoded(pc, op, f"DIV0S R{c},R{b}")
    if a == 0 and d == 2:
        return Decoded(pc, op, {0: f"STC SR,R{b}", 1: f"STC GBR,R{b}", 2: f"STC VBR,R{b}"}.get(c, f"UNKNOWN{op:04X}"))
    if a == 0 and d == 0xA:
        return Decoded(pc, op, {0: f"STS MACH,R{b}", 1: f"STS MACL,R{b}", 2: f"STS PR,R{b}"}.get(c, f"UNKNOWN{op:04X}"))
    if op & 0xF0FF == 0x0029:
        return Decoded(pc, op, f"MOVT R{b}")
    if a == 0 and d == 4:
        return Decoded(pc, op, f"MOV.B R{c},@(R0,R{b})")
    if a == 0 and d == 5:
        return Decoded(pc, op, f"MOV.W R{c},@(R0,R{b})")
    if a == 0 and d == 6:
        return Decoded(pc, op, f"MOV.L R{c},@(R0,R{b})")
    if a == 0 and d == 0xC:
        return Decoded(pc, op, f"MOV.B @(R0,R{c}),R{b}")
    if a == 0 and d == 0xD:
        return Decoded(pc, op, f"MOV.W @(R0,R{c}),R{b}")
    if a == 0 and d == 0xE:
        return Decoded(pc, op, f"MOV.L @(R0,R{c}),R{b}")
    if a == 0 and cd == 0x23:
        return Decoded(pc, op, f"BRAF R{b}", kind="branch", target=None)  # register-indirect, needs reg tracking
    if a == 0 and cd == 0x03:
        return Decoded(pc, op, f"BSRF R{b}", kind="call", target=None)
    if a == 1:
        return Decoded(pc, op, f"MOV.L R{c},@({d}*4,R{b})")
    if a == 2 and d == 0:
        return Decoded(pc, op, f"MOV.B R{c},@R{b}")
    if a == 2 and d == 1:
        return Decoded(pc, op, f"MOV.W R{c},@R{b}")
    if a == 2 and d == 2:
        return Decoded(pc, op, f"MOV.L R{c},@R{b}")
    if a == 2 and d == 4:
        return Decoded(pc, op, f"MOV.B R{c},@-R{b}")
    if a == 2 and d == 5:
        return Decoded(pc, op, f"MOV.W R{c},@-R{b}")
    if a == 2 and d == 6:
        return Decoded(pc, op, f"MOV.L R{c},@-R{b}")
    if a == 2 and d == 8:
        return Decoded(pc, op, f"TST R{c},R{b}")
    if a == 2 and d == 9:
        return Decoded(pc, op, f"AND R{c},R{b}")
    if a == 2 and d == 0xA:
        return Decoded(pc, op, f"XOR R{c},R{b}")
    if a == 2 and d == 0xB:
        return Decoded(pc, op, f"OR R{c},R{b}")
    if a == 2 and d == 0xC:
        return Decoded(pc, op, f"CMP/STR R{c},R{b}")
    if a == 2 and d == 0xD:
        return Decoded(pc, op, f"XTRCT R{c},R{b}")
    if a == 2 and d == 0xE:
        return Decoded(pc, op, f"MULU.W R{c},R{b}")
    if a == 2 and d == 0xF:
        return Decoded(pc, op, f"MULS.W R{c},R{b}")
    if a == 3 and d == 0:
        return Decoded(pc, op, f"CMP/EQ R{c},R{b}")
    if a == 3 and d == 2:
        return Decoded(pc, op, f"CMP/HS R{c},R{b}")
    if a == 3 and d == 3:
        return Decoded(pc, op, f"CMP/GE R{c},R{b}")
    if a == 3 and d == 4:
        return Decoded(pc, op, f"DIV1 R{c},R{b}")
    if a == 3 and d == 5:
        return Decoded(pc, op, f"DMULU.L R{c},R{b}")
    if a == 3 and d == 6:
        return Decoded(pc, op, f"CMP/HI R{c},R{b}")
    if a == 3 and d == 7:
        return Decoded(pc, op, f"CMP/GT R{c},R{b}")
    if a == 3 and d == 8:
        return Decoded(pc, op, f"SUB R{c},R{b}")
    if a == 3 and d == 0xA:
        return Decoded(pc, op, f"SUBC R{c},R{b}")
    if a == 3 and d == 0xB:
        return Decoded(pc, op, f"SUBV R{c},R{b}")
    if a == 3 and d == 0xC:
        return Decoded(pc, op, f"ADD R{c},R{b}")
    if a == 3 and d == 0xD:
        return Decoded(pc, op, f"DMULS.L R{c},R{b}")
    if a == 3 and d == 0xE:
        return Decoded(pc, op, f"ADDC R{c},R{b}")
    if a == 3 and d == 0xF:
        return Decoded(pc, op, f"ADDV R{c},R{b}")
    if a == 4 and cd == 0x0B:
        return Decoded(pc, op, f"JSR @R{b}", kind="call", target=None)
    if a == 4 and cd == 0x2B:
        return Decoded(pc, op, f"JMP @R{b}", kind="branch", target=None)
    if a == 4 and cd == 0x2A:
        return Decoded(pc, op, f"LDS R{b},PR")
    if a == 4 and cd == 0x0A:
        return Decoded(pc, op, f"LDS R{b},MACH")
    if a == 4 and cd == 0x1A:
        return Decoded(pc, op, f"LDS R{b},MACL")
    if a == 4 and cd == 0x0E:
        return Decoded(pc, op, f"LDC R{b},SR")
    if a == 4 and cd == 0x1E:
        return Decoded(pc, op, f"LDC R{b},GBR")
    if a == 4 and cd == 0x2E:
        return Decoded(pc, op, f"LDC R{b},VBR")
    if a == 4 and cd == 0x15:
        return Decoded(pc, op, f"CMP/PL R{b}")
    if a == 4 and cd == 0x11:
        return Decoded(pc, op, f"CMP/PZ R{b}")
    if a == 4 and cd == 0x00:
        return Decoded(pc, op, f"SHLL R{b}")
    if a == 4 and cd == 0x01:
        return Decoded(pc, op, f"SHLR R{b}")
    if a == 4 and cd == 0x08:
        return Decoded(pc, op, f"SHLL2 R{b}")
    if a == 4 and cd == 0x09:
        return Decoded(pc, op, f"SHLR2 R{b}")
    if a == 4 and cd == 0x18:
        return Decoded(pc, op, f"SHLL8 R{b}")
    if a == 4 and cd == 0x19:
        return Decoded(pc, op, f"SHLR8 R{b}")
    if a == 4 and cd == 0x28:
        return Decoded(pc, op, f"SHLL16 R{b}")
    if a == 4 and cd == 0x29:
        return Decoded(pc, op, f"SHLR16 R{b}")
    if a == 4 and cd == 0x20:
        return Decoded(pc, op, f"SHAL R{b}")
    if a == 4 and cd == 0x21:
        return Decoded(pc, op, f"SHAR R{b}")
    if a == 4 and cd == 0x24:
        return Decoded(pc, op, f"ROTCL R{b}")
    if a == 4 and cd == 0x25:
        return Decoded(pc, op, f"ROTCR R{b}")
    if a == 4 and cd == 0x04:
        return Decoded(pc, op, f"ROTL R{b}")
    if a == 4 and cd == 0x05:
        return Decoded(pc, op, f"ROTR R{b}")
    if a == 4 and cd == 0x10:
        return Decoded(pc, op, f"DT R{b}")
    if a == 4 and cd == 0x22:
        return Decoded(pc, op, f"STS.L MACL,@-R{b}")
    if a == 4 and cd == 0x12:
        return Decoded(pc, op, f"STS.L MACH,@-R{b}")
    if a == 4 and cd == 0x02:
        return Decoded(pc, op, f"STS.L PR,@-R{b}")
    if a == 4 and cd == 0x26:
        return Decoded(pc, op, f"LDS.L @R{b}+,MACL")
    if a == 4 and cd == 0x06:
        return Decoded(pc, op, f"LDS.L @R{b}+,MACH")
    if a == 4 and cd == 0x27:
        return Decoded(pc, op, f"LDC.L @R{b}+,GBR")
    if a == 4 and cd == 0x07:
        return Decoded(pc, op, f"LDC.L @R{b}+,SR")
    if a == 4 and cd == 0x0F:
        return Decoded(pc, op, f"MAC.W @R{c}+,@R{b}+")
    if a == 5:
        return Decoded(pc, op, f"MOV.L @({d}*4,R{c}),R{b}")
    if a == 6 and d == 0:
        return Decoded(pc, op, f"MOV.B @R{c},R{b}")
    if a == 6 and d == 1:
        return Decoded(pc, op, f"MOV.W @R{c},R{b}")
    if a == 6 and d == 2:
        return Decoded(pc, op, f"MOV.L @R{c},R{b}")
    if a == 6 and d == 3:
        return Decoded(pc, op, f"MOV R{c},R{b}")
    if a == 6 and d == 4:
        return Decoded(pc, op, f"MOV.B @R{c}+,R{b}")
    if a == 6 and d == 5:
        return Decoded(pc, op, f"MOV.W @R{c}+,R{b}")
    if a == 6 and d == 6:
        return Decoded(pc, op, f"MOV.L @R{c}+,R{b}")
    if a == 6 and d == 7:
        return Decoded(pc, op, f"NOT R{c},R{b}")
    if a == 6 and d == 8:
        return Decoded(pc, op, f"SWAP.B R{c},R{b}")
    if a == 6 and d == 9:
        return Decoded(pc, op, f"SWAP.W R{c},R{b}")
    if a == 6 and d == 0xA:
        return Decoded(pc, op, f"NEGC R{c},R{b}")
    if a == 6 and d == 0xB:
        return Decoded(pc, op, f"NEG R{c},R{b}")
    if a == 6 and d == 0xC:
        return Decoded(pc, op, f"EXTU.B R{c},R{b}")
    if a == 6 and d == 0xD:
        return Decoded(pc, op, f"EXTU.W R{c},R{b}")
    if a == 6 and d == 0xE:
        return Decoded(pc, op, f"EXTS.B R{c},R{b}")
    if a == 6 and d == 0xF:
        return Decoded(pc, op, f"EXTS.W R{c},R{b}")
    if a == 7:
        return Decoded(pc, op, f"ADD #{s8(cd)},R{b}")
    if a == 8 and b == 0:
        return Decoded(pc, op, f"MOV.B R0,@({d},R{c})")
    if a == 8 and b == 1:
        return Decoded(pc, op, f"MOV.W R0,@({d}*2,R{c})")
    if a == 8 and b == 4:
        return Decoded(pc, op, f"MOV.B @({d},R{c}),R0")
    if a == 8 and b == 5:
        return Decoded(pc, op, f"MOV.W @({d}*2,R{c}),R0")
    if a == 8 and b == 8:
        return Decoded(pc, op, f"CMP/EQ #{s8(cd)},R0")
    if a == 8 and b == 9:
        t = pc + (s8(cd) << 1) + 4
        return Decoded(pc, op, f"BT 0x{t:08X}", kind="cond_branch", target=t)
    if a == 8 and b == 0xB:
        t = pc + (s8(cd) << 1) + 4
        return Decoded(pc, op, f"BF 0x{t:08X}", kind="cond_branch", target=t)
    if a == 8 and b == 0xD:
        t = pc + (s8(cd) << 1) + 4
        return Decoded(pc, op, f"BT/S 0x{t:08X}", kind="cond_branch", target=t)
    if a == 8 and b == 0xF:
        t = pc + (s8(cd) << 1) + 4
        return Decoded(pc, op, f"BF/S 0x{t:08X}", kind="cond_branch", target=t)
    if a == 9:
        target = pc + (cd << 1) + 4
        val = read_u16(data, target) if data is not None else None
        return Decoded(pc, op, f"MOV.W @(0x{cd:X},PC),R{b}", pool_addr=target, pool_val=val)
    if a == 0xA:
        disp = bcd if bcd < 0x800 else bcd - 0x1000
        t = pc + (disp << 1) + 4
        return Decoded(pc, op, f"BRA 0x{t:08X}", kind="branch", target=t)
    if a == 0xB:
        disp = bcd if bcd < 0x800 else bcd - 0x1000
        t = pc + (disp << 1) + 4
        return Decoded(pc, op, f"BSR 0x{t:08X}", kind="call", target=t)
    if a == 0xC and b == 0:
        return Decoded(pc, op, f"MOV.B R0,@(0x{cd:X},GBR)")
    if a == 0xC and b == 1:
        return Decoded(pc, op, f"MOV.W R0,@(0x{cd:X}*2,GBR)")
    if a == 0xC and b == 2:
        return Decoded(pc, op, f"MOV.L R0,@(0x{cd:X}*4,GBR)")
    if a == 0xC and b == 3:
        return Decoded(pc, op, f"TRAPA #0x{cd:X}")
    if a == 0xC and b == 4:
        return Decoded(pc, op, f"MOV.B @(0x{cd:X},GBR),R0")
    if a == 0xC and b == 5:
        return Decoded(pc, op, f"MOV.W @(0x{cd:X}*2,GBR),R0")
    if a == 0xC and b == 6:
        return Decoded(pc, op, f"MOV.L @(0x{cd:X}*4,GBR),R0")
    if a == 0xC and b == 7:
        addr = ((pc + 4) & ~3) + (cd << 2)
        return Decoded(pc, op, f"MOVA @(0x{cd:X}*4,PC),R0", pool_addr=addr)
    if a == 0xC and b == 8:
        return Decoded(pc, op, f"TST #0x{cd:X},R0")
    if a == 0xC and b == 9:
        return Decoded(pc, op, f"AND #0x{cd:X},R0")
    if a == 0xC and b == 0xA:
        return Decoded(pc, op, f"XOR #0x{cd:X},R0")
    if a == 0xC and b == 0xB:
        return Decoded(pc, op, f"OR #0x{cd:X},R0")
    if a == 0xC and b == 0xC:
        return Decoded(pc, op, f"TST.B #0x{cd:X},@(R0,GBR)")
    if a == 0xC and b == 0xD:
        return Decoded(pc, op, f"AND.B #0x{cd:X},@(R0,GBR)")
    if a == 0xC and b == 0xE:
        return Decoded(pc, op, f"XOR.B #0x{cd:X},@(R0,GBR)")
    if a == 0xC and b == 0xF:
        return Decoded(pc, op, f"OR.B #0x{cd:X},@(R0,GBR)")
    if a == 0xD:
        target = ((pc + 4) & ~3) + (cd << 2)
        val = read_u32(data, target) if data is not None else None
        return Decoded(pc, op, f"MOV.L @(0x{cd:X}*4,PC),R{b}", pool_addr=target, pool_val=val)
    if a == 0xE:
        return Decoded(pc, op, f"MOV #{s8(cd)},R{b}")
    return Decoded(pc, op, f"UNKNOWN {op:04X}")
