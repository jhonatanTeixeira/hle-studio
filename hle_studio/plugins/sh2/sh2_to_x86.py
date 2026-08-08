#!/usr/bin/env python3
"""Mechanical SH-2 -> x86_64 (Intel syntax, System V ABI) instruction
translator, used by study_call.py to append a "what would this look like
recompiled" section to each function study.

This is deliberately NOT a real register allocator or code generator - it's
a fixed, documented, one-instruction-at-a-time transliteration, same spirit
as the rest of this project's tooling: automate the mechanical part, flag
what isn't modeled instead of guessing (see GEMINI.md).

=== Register convention ===

SH-2 has exactly 16 general registers (R0-R15) and x86_64 has exactly 16
general-purpose registers - a clean 1:1 mapping, chosen for maximum
*semantic* correspondence rather than arbitrary order:

    R0  -> eax   (SH-2's de facto return-value register; x86 return register too)
    R4  -> edi   (arg1, matches SysV's arg1)
    R5  -> esi   (arg2)
    R6  -> edx   (arg3)
    R7  -> ecx   (arg4)
    R14 -> ebp   (observed used as a preserved-across-calls/frame-ish register,
                  matching ebp's conventional role)
    R15 -> esp   (R15 IS the SH-2 hardware stack pointer - exact architectural
                  match. This makes MOV.L @-R15/@R15+ translate directly to
                  push/pop, not a manual increment/decrement.)
    R1,R2,R3,R8,R9,R10,R11,R12,R13 -> r8d,r9d,r10d,r11d,r12d,r13d,r14d,r15d,ebx
                  (remaining scratch registers, arbitrary but fixed assignment)

PR (the link register) needs no dedicated host register: x86_64 `call`/`ret`
push/pop the return address on the stack natively, which is strictly better
than SH-2's single-PR-register scheme (no need to spill PR before a nested
call - `call` on x86 already does the equivalent of "save old PR").

MACH:MACL (the 32+32-bit multiply/accumulate pair) has no fixed host home -
all 16 SH-2 registers already claim all 16 x86-64 GPRs, so there is no truly
"spare" register. The natural choice is RDX:RAX (x86's own hardware-defined
widening-multiply output), which is what these translations use - but note
this clobbers whatever SH-2 registers are mapped to eax/edx (R0/R6) if they
are live across the multiply. A real recompiler resolves this with spill/
reload; these translations flag it rather than hide it.

Saturn memory is big-endian; x86_64 is little-endian. Every multi-byte load/
store uses `movbe` (a real x86_64 instruction for exactly this) rather than
plain `mov` - see CLAUDE.md's own endianness rule. The emulated address
space is assumed reachable via a fixed base symbol `saturn_mem` (this
project's SaturnMemory is a single global structure, so a RIP-relative fixed
symbol is realistic - no per-call base-pointer argument needed).

Accesses that fall inside a known hardware-mapped region (docs/
memory_map_plan.md) are translated as a call to a named HLE helper instead
of a raw load/store - that recognition is the actual point of this whole
project. Where only the *region* is known (not a specific named register),
a generic `call read_u16_<region>` placeholder is used instead of inventing
semantics.
"""
import re

REG_MAP = {
    0: "eax", 1: "r8d", 2: "r9d", 3: "r10d",
    4: "edi", 5: "esi", 6: "edx", 7: "ecx",
    8: "r11d", 9: "r12d", 10: "r13d", 11: "r14d",
    12: "r15d", 13: "ebx", 14: "ebp", 15: "esp",
}
REG_MAP64 = {n: ("r" + v[1:] if v.startswith("e") else v.replace("d", "", 1) if v.startswith("r") and v.endswith("d") else v)
             for n, v in REG_MAP.items()}
# Explicit overrides where the mechanical rename above doesn't land right:
REG_MAP64 = {
    0: "rax", 1: "r8", 2: "r9", 3: "r10",
    4: "rdi", 5: "rsi", 6: "rdx", 7: "rcx",
    8: "r11", 9: "r12", 10: "r13", 11: "r14",
    12: "r15", 13: "rbx", 14: "rbp", 15: "rsp",
}
REG_MAP16 = {
    0: "ax", 1: "r8w", 2: "r9w", 3: "r10w",
    4: "di", 5: "si", 6: "dx", 7: "cx",
    8: "r11w", 9: "r12w", 10: "r13w", 11: "r14w",
    12: "r15w", 13: "bx", 14: "bp", 15: "sp",
}
REG_MAP8 = {
    0: "al", 1: "r8b", 2: "r9b", 3: "r10b",
    4: "dil", 5: "sil", 6: "dl", 7: "cl",
    8: "r11b", 9: "r12b", 10: "r13b", 11: "r14b",
    12: "r15b", 13: "bl", 14: "bpl", 15: "spl",
}

# Known hardware regions (docs/memory_map_plan.md), (low, high, name) - addr
# masked with 0x0FFFFFFF first to fold both the 0x0xxxxxxx cache-enabled and
# 0x2xxxxxxx cache-through mirrors onto the same range (see address_mapping.md).
REGIONS = [
    (0x00000000, 0x000FFFFF, "BIOS ROM"),
    (0x00100000, 0x0017FFFF, "SMPC"),
    (0x00180000, 0x001FFFFF, "Backup RAM"),
    (0x00200000, 0x002FFFFF, "Low WRAM"),
    (0x02000000, 0x03FFFFFF, "CS0"),
    (0x04000000, 0x04FFFFFF, "CS1"),
    (0x05800000, 0x058FFFFF, "CS2 (bloco de CD)"),
    (0x05A00000, 0x05AFFFFF, "SCSP wave RAM"),
    (0x05B00000, 0x05BFFFFF, "SCSP registradores"),
    (0x05C00000, 0x05C7FFFF, "VDP1 VRAM"),
    (0x05C80000, 0x05CFFFFF, "VDP1 Framebuffer"),
    (0x05D00000, 0x05D7FFFF, "VDP1 registradores"),
    (0x05E00000, 0x05EFFFFF, "VDP2 VRAM"),
    (0x05F00000, 0x05F7FFFF, "VDP2 Color RAM"),
    (0x05F80000, 0x05FBFFFF, "VDP2 registradores"),
    (0x05FE0000, 0x05FEFFFF, "SCU registradores"),
    (0x06000000, 0x0610FFFF, "High WRAM"),
]

# Specific, individually-identified hardware registers (grows as more get
# studied - see tools/function_catalog.json for the prose behind each).
KNOWN_REGISTERS = {
    0x05890008: "read_hirq",  # CS2 HIRQ, confirmed against vendor/yabause/src/cs2.c
}


def region_for(addr):
    a = addr & 0x0FFFFFFF
    for lo, hi, name in REGIONS:
        if lo <= a <= hi:
            return name
    return None


def s8(v):
    return v - 0x100 if v & 0x80 else v


def r32(n):
    return REG_MAP[n]


def r64(n):
    return REG_MAP64[n]


def r16(n):
    return REG_MAP16[n]


def r8(n):
    return REG_MAP8[n]


class Translator:
    """Tracks, per straight-line function body, which registers currently
    hold a statically-known constant (constant propagation, same spirit as
    study_frame.py's RegTracker) - just enough to recognize "this @Rn access
    targets a known hardware register" the way we did by hand for 060B780E."""

    def __init__(self):
        self.const = {i: None for i in range(16)}

    def set_const(self, n, v):
        self.const[n] = v & 0xFFFFFFFF if v is not None else None

    def invalidate(self, n):
        self.const[n] = None

    def get_const(self, n):
        return self.const.get(n)

    def hw_target(self, base_reg):
        """If base_reg currently holds a known constant address, return
        (region_name_or_None, helper_name_or_None) - both None means "not a
        recognized hardware address, translate as a normal memory access"."""
        v = self.get_const(base_reg)
        if v is None:
            return None, None
        masked = v & 0x0FFFFFFF
        helper = KNOWN_REGISTERS.get(masked)
        region = region_for(v)
        return region, helper


def mem_operand(base_expr, size_kw=None):
    pfx = f"{size_kw} " if size_kw else ""
    return f"{pfx}[saturn_mem + {base_expr}]"


def translate_one(dec, tr, is_delay_slot=False):
    """Translate one decoded SH-2 instruction to a list of x86_64 asm lines
    (Intel syntax). `tr` is the Translator (const-tracking) state, already
    updated by the CALLER for instructions *before* this one - this function
    both emits code and mutates `tr` for this instruction's effect, mirroring
    RegTracker.apply()'s single-pass style in study_frame.py."""
    m = dec.mnemonic
    lines = []
    note = "   ; (delay slot)" if is_delay_slot else ""

    def emit(*asm):
        lines.extend(asm)

    # --- pool loads (already resolved to a static constant from 0.BIN) ---
    if dec.pool_addr is not None and m.startswith("MOV.L @(0x"):
        n = int(re.search(r",R(\d+)$", m).group(1))
        val = dec.pool_val
        emit(f"mov {r32(n)}, 0x{val:08X}{note}" if val is not None else f"; pool não resolvido{note}")
        tr.set_const(n, val)
        return lines
    if dec.pool_addr is not None and m.startswith("MOV.W @(0x") and "PC)" in m:
        n = int(re.search(r",R(\d+)$", m).group(1))
        val = dec.pool_val
        if val is not None:
            sval = (val - 0x10000) if val & 0x8000 else val
            emit(f"mov {r32(n)}, 0x{sval & 0xFFFFFFFF:08X}   ; sign-extend de word 0x{val:04X}{note}")
            tr.set_const(n, sval & 0xFFFFFFFF)
        else:
            emit(f"; pool não resolvido{note}")
        return lines
    if dec.pool_addr is not None and m.startswith("MOVA"):
        emit(f"mov {r32(0)}, 0x{dec.pool_addr:08X}   ; endereço do pool, não seu conteúdo{note}")
        tr.set_const(0, dec.pool_addr)
        return lines

    # --- register moves / immediates ---
    mo = re.match(r"^MOV R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"mov {r32(rn)}, {r32(rm)}{note}")
        tr.set_const(rn, tr.get_const(rm))
        return lines
    mo = re.match(r"^MOV #(-?\d+),R(\d+)$", m)
    if mo:
        imm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"mov {r32(rn)}, {imm}{note}")
        tr.set_const(rn, imm & 0xFFFFFFFF)
        return lines

    # --- stack push/pop: R15 IS esp, so these are literal push/pop ---
    mo = re.match(r"^MOV\.L R(\d+),@-R15$", m)
    if mo:
        rn = int(mo.group(1))
        emit(f"push {r64(rn)}   ; R15 == rsp, pré-decremento vira push nativo{note}")
        return lines
    mo = re.match(r"^MOV\.L @R15\+,R(\d+)$", m)
    if mo:
        rn = int(mo.group(1))
        emit(f"pop {r64(rn)}   ; R15 == rsp, pós-incremento vira pop nativo{note}")
        tr.invalidate(rn)
        return lines
    if m == "STS.L MACL,@-R15":
        emit(f"push rax   ; MACL ~ rax (ver convenção MACH:MACL no topo do arquivo){note}")
        return lines
    if m == "LDS.L @R15+,MACL":
        emit(f"pop rax   ; restaura MACL ~ rax{note}")
        return lines
    mo = re.match(r"^MOV\.L R(\d+),@-R(\d+)$", m)
    if mo:
        rn, rm = int(mo.group(1)), int(mo.group(2))
        emit(f"sub {r64(rm)}, 4", f"movbe {mem_operand(r64(rm), 'dword')}, {r32(rn)}{note}")
        return lines
    mo = re.match(r"^MOV\.L @R(\d+)\+,R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"movbe {r32(rn)}, {mem_operand(r64(rm), 'dword')}", f"add {r64(rm)}, 4{note}")
        tr.invalidate(rn)
        return lines

    # --- plain register-indirect load/store (@Rm) ---
    mo = re.match(r"^MOV\.([BWL]) @R(\d+),R(\d+)$", m)
    if mo:
        width, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        region, helper = tr.hw_target(rm)
        if helper:
            emit(f"call {helper}   ; {r32(rm)} == 0x{tr.get_const(rm):08X}, registrador conhecido{note}",
                 f"mov {r32(rn)}, eax   ; valor de retorno do helper")
            tr.invalidate(rn)
            return lines
        if region:
            emit(f"; ATENÇÃO: {r32(rm)} aponta para a região \"{region}\" mas o registrador exato "
                 f"ainda não foi identificado — ver docs/memory_map_plan.md antes de tratar isso como WRAM comum",
                 f"call read_u{ {'B':8,'W':16,'L':32}[width] }_{region.split()[0].lower()}   ; placeholder{note}")
            tr.invalidate(rn)
            return lines
        if width == "B":
            emit(f"movsx {r32(rn)}, byte {mem_operand(r64(rm))}{note}")
        elif width == "W":
            emit(f"movbe {r16(rn)}, word {mem_operand(r64(rm))}",
                 f"movsx {r32(rn)}, {r16(rn)}   ; MOV.W sign-extends{note}")
        else:
            emit(f"movbe {r32(rn)}, {mem_operand(r64(rm), 'dword')}{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^MOV\.([BWL]) R(\d+),@R(\d+)$", m)
    if mo:
        width, rn, rm = mo.group(1), int(mo.group(2)), int(mo.group(3))
        region, helper = tr.hw_target(rm)
        if helper or region:
            target = helper or f"write_u{ {'B':8,'W':16,'L':32}[width] }_{region.split()[0].lower()}"
            emit(f"mov edi, {r32(rn)}   ; arg1 = valor a escrever", f"call {target}{note}")
            return lines
        reg = {"B": r8(rn), "W": r16(rn), "L": r32(rn)}[width]
        kw = {"B": None, "W": "word", "L": "dword"}[width]
        instr = "mov" if width == "B" else "movbe"
        emit(f"{instr} {mem_operand(r64(rm), kw)}, {reg}{note}")
        return lines

    # --- indexed by R0: @(R0,Rm) ---
    mo = re.match(r"^MOV\.([BWL]) @\(R0,R(\d+)\),R(\d+)$", m)
    if mo:
        width, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        addr = f"{r64(0)} + {r64(rm)}"
        if width == "B":
            emit(f"movsx {r32(rn)}, byte {mem_operand(addr)}{note}")
        elif width == "W":
            emit(f"movbe {r16(rn)}, word {mem_operand(addr)}", f"movsx {r32(rn)}, {r16(rn)}{note}")
        else:
            emit(f"movbe {r32(rn)}, {mem_operand(addr, 'dword')}{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^MOV\.([BWL]) R(\d+),@\(R0,R(\d+)\)$", m)
    if mo:
        width, rn, rm = mo.group(1), int(mo.group(2)), int(mo.group(3))
        addr = f"{r64(0)} + {r64(rm)}"
        reg = {"B": r8(rn), "W": r16(rn), "L": r32(rn)}[width]
        kw = {"B": None, "W": "word", "L": "dword"}[width]
        instr = "mov" if width == "B" else "movbe"
        emit(f"{instr} {mem_operand(addr, kw)}, {reg}{note}")
        return lines

    # --- displacement forms: @(disp,Rm)/@(disp*2,Rm)/@(disp*4,Rm), R0 or Rn ---
    mo = re.match(r"^MOV\.([BW]) @\((\d+)(?:\*2)?,R(\d+)\),R0$", m)
    if mo:
        width, disp, rm = mo.group(1), int(mo.group(2)), int(mo.group(3))
        addr = f"{r64(rm)} + {disp * (2 if width == 'W' else 1)}"
        if width == "B":
            emit(f"movsx eax, byte {mem_operand(addr)}{note}")
        else:
            emit(f"movbe ax, word {mem_operand(addr)}", f"movsx eax, ax{note}")
        tr.invalidate(0)
        return lines
    mo = re.match(r"^MOV\.([BW]) R0,@\((\d+)(?:\*2)?,R(\d+)\)$", m)
    if mo:
        width, disp, rm = mo.group(1), int(mo.group(2)), int(mo.group(3))
        addr = f"{r64(rm)} + {disp * (2 if width == 'W' else 1)}"
        if width == "B":
            emit(f"mov {mem_operand(addr)}, al{note}")
        else:
            emit(f"movbe {mem_operand(addr, 'word')}, ax{note}")
        return lines
    mo = re.match(r"^MOV\.L @\((\d+)\*4,R(\d+)\),R(\d+)$", m)
    if mo:
        disp, rm, rn = int(mo.group(1)), int(mo.group(2)), int(mo.group(3))
        emit(f"movbe {r32(rn)}, {mem_operand(f'{r64(rm)} + {disp * 4}', 'dword')}{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^MOV\.L R(\d+),@\((\d+)\*4,R(\d+)\)$", m)
    if mo:
        rn, disp, rm = int(mo.group(1)), int(mo.group(2)), int(mo.group(3))
        emit(f"movbe {mem_operand(f'{r64(rm)} + {disp * 4}', 'dword')}, {r32(rn)}{note}")
        return lines
    mo = re.match(r"^MOV\.W @\((R0,)?R(\d+)\),R(\d+)$", m)
    if mo:
        # already covered by @(R0,Rm) pattern above; kept for completeness
        pass

    # --- arithmetic ---
    mo = re.match(r"^ADD R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"add {r32(rn)}, {r32(rm)}{note}")
        a, b = tr.get_const(rm), tr.get_const(rn)
        tr.set_const(rn, (a + b) if (a is not None and b is not None) else None)
        return lines
    mo = re.match(r"^ADD #(-?\d+),R(\d+)$", m)
    if mo:
        imm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"add {r32(rn)}, {imm}{note}")
        cur = tr.get_const(rn)
        tr.set_const(rn, (cur + imm) if cur is not None else None)
        return lines
    mo = re.match(r"^ADDC R(\d+),R(\d+)$", m)
    if mo:
        emit(f"adc {r32(int(mo.group(2)))}, {r32(int(mo.group(1)))}   ; T = carry-out, igual ao CF do x86{note}")
        tr.invalidate(int(mo.group(2)))
        return lines
    mo = re.match(r"^ADDV R(\d+),R(\d+)$", m)
    if mo:
        emit(f"add {r32(int(mo.group(2)))}, {r32(int(mo.group(1)))}   ; T = overflow, ver OF (seto) se precisar{note}")
        tr.invalidate(int(mo.group(2)))
        return lines
    mo = re.match(r"^SUB R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"sub {r32(rn)}, {r32(rm)}{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^SUBC R(\d+),R(\d+)$", m)
    if mo:
        emit(f"sbb {r32(int(mo.group(2)))}, {r32(int(mo.group(1)))}   ; T = borrow-out{note}")
        tr.invalidate(int(mo.group(2)))
        return lines
    mo = re.match(r"^SUBV R(\d+),R(\d+)$", m)
    if mo:
        emit(f"sub {r32(int(mo.group(2)))}, {r32(int(mo.group(1)))}   ; T = overflow{note}")
        tr.invalidate(int(mo.group(2)))
        return lines
    mo = re.match(r"^NEG R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        if rm != rn:
            emit(f"mov {r32(rn)}, {r32(rm)}")
        emit(f"neg {r32(rn)}{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^NEGC R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"mov {r32(rn)}, 0", f"sbb {r32(rn)}, {r32(rm)}   ; T = borrow-out{note}")
        tr.invalidate(rn)
        return lines

    # --- logic ---
    for op, x86 in (("AND", "and"), ("OR", "or"), ("XOR", "xor")):
        mo = re.match(rf"^{op} R(\d+),R(\d+)$", m)
        if mo:
            rm, rn = int(mo.group(1)), int(mo.group(2))
            emit(f"{x86} {r32(rn)}, {r32(rm)}{note}")
            tr.invalidate(rn)
            return lines
    for op, x86 in (("AND", "and"), ("OR", "or"), ("XOR", "xor"), ("TST", "test")):
        mo = re.match(rf"^{op} #0x([0-9A-F]+),R0$", m)
        if mo:
            imm = int(mo.group(1), 16)
            emit(f"{x86} eax, 0x{imm:X}{note}")
            if op != "TST":
                tr.invalidate(0)
            else:
                tr.pending_cc = ("e", "ne")  # T=1 <=> ZF=1 (AND deu zero) - T bit's own meaning, matches ZF directly
            return lines
    mo = re.match(r"^TST R(\d+),R(\d+)$", m)
    if mo:
        emit(f"test {r32(int(mo.group(2)))}, {r32(int(mo.group(1)))}   ; T=1 <=> ZF=1 (AND deu zero){note}")
        tr.pending_cc = ("e", "ne")
        return lines
    mo = re.match(r"^NOT R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        if rm != rn:
            emit(f"mov {r32(rn)}, {r32(rm)}")
        emit(f"not {r32(rn)}{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^XTRCT R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"shrd {r32(rn)}, {r32(rm)}, 16   ; XTRCT: metade alta de Rn + metade baixa de Rm{note}")
        tr.invalidate(rn)
        return lines

    # --- shifts / rotates ---
    for name, cnt in (("SHLL2", 2), ("SHLR2", 2), ("SHLL8", 8), ("SHLR8", 8), ("SHLL16", 16), ("SHLR16", 16)):
        mo = re.match(rf"^{name} R(\d+)$", m)
        if mo:
            rn = int(mo.group(1))
            x86 = "shl" if "SHLL" in name else "shr"
            emit(f"{x86} {r32(rn)}, {cnt}{note}")
            cur = tr.get_const(rn)
            if cur is not None:
                tr.set_const(rn, (cur << cnt) if x86 == "shl" else (cur >> cnt))
            return lines
    mo = re.match(r"^SHLL R(\d+)$", m)
    if mo:
        rn = int(mo.group(1))
        emit(f"shl {r32(rn)}, 1{note}")
        cur = tr.get_const(rn)
        tr.set_const(rn, (cur << 1) if cur is not None else None)
        return lines
    mo = re.match(r"^SHLR R(\d+)$", m)
    if mo:
        rn = int(mo.group(1))
        emit(f"shr {r32(rn)}, 1{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^SHAR R(\d+)$", m)
    if mo:
        rn = int(mo.group(1))
        emit(f"sar {r32(rn)}, 1{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^SHAL R(\d+)$", m)
    if mo:
        rn = int(mo.group(1))
        emit(f"sal {r32(rn)}, 1{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^ROTL R(\d+)$", m)
    if mo:
        emit(f"rol {r32(int(mo.group(1)))}, 1{note}")
        tr.invalidate(int(mo.group(1)))
        return lines
    mo = re.match(r"^ROTR R(\d+)$", m)
    if mo:
        emit(f"ror {r32(int(mo.group(1)))}, 1{note}")
        tr.invalidate(int(mo.group(1)))
        return lines
    mo = re.match(r"^ROTCL R(\d+)$", m)
    if mo:
        emit(f"rcl {r32(int(mo.group(1)))}, 1   ; usa T/CF como carry-in, igual ao x86{note}")
        tr.invalidate(int(mo.group(1)))
        return lines
    mo = re.match(r"^ROTCR R(\d+)$", m)
    if mo:
        emit(f"rcr {r32(int(mo.group(1)))}, 1{note}")
        tr.invalidate(int(mo.group(1)))
        return lines

    # --- compares (fold into the NEXT BT/BF's jcc - see study_call.py caller) ---
    CMP_CC = {
        "CMP/EQ": ("e", "ne"), "CMP/HS": ("ae", "b"), "CMP/GE": ("ge", "l"),
        "CMP/HI": ("a", "be"), "CMP/GT": ("g", "le"),
    }
    mo = re.match(r"^(CMP/[A-Z]+) R(\d+),R(\d+)$", m)
    if mo and mo.group(1) in CMP_CC:
        op, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        emit(f"cmp {r32(rn)}, {r32(rm)}   ; T definido para o próximo BT/BF, ver abaixo{note}")
        tr.pending_cc = CMP_CC[op]
        return lines
    mo = re.match(r"^CMP/EQ #(-?\d+),R0$", m)
    if mo:
        emit(f"cmp eax, {int(mo.group(1))}{note}")
        tr.pending_cc = CMP_CC["CMP/EQ"]
        return lines
    mo = re.match(r"^CMP/PZ R(\d+)$", m)
    if mo:
        emit(f"test {r32(int(mo.group(1)))}, {r32(int(mo.group(1)))}   ; PZ: >=0{note}")
        tr.pending_cc = ("ns", "s")
        return lines
    mo = re.match(r"^CMP/PL R(\d+)$", m)
    if mo:
        emit(f"test {r32(int(mo.group(1)))}, {r32(int(mo.group(1)))}   ; PL: >0{note}")
        tr.pending_cc = ("g", "le")
        return lines
    mo = re.match(r"^DT R(\d+)$", m)
    if mo:
        emit(f"dec {r32(int(mo.group(1)))}   ; T=1 se deu zero{note}")
        tr.pending_cc = ("e", "ne")
        return lines

    # --- MAC / multiply (see module docstring re: RDX:RAX pressure) ---
    mo = re.match(r"^DMULS\.L R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"movsxd rax, {r32(rm)}", f"movsxd rdx, {r32(rn)}",
             f"imul rax, rdx   ; produto 64-bit assinado -> \"MACL:MACH\" ~ rax{note}")
        return lines
    mo = re.match(r"^DMULU\.L R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"mov eax, {r32(rm)}", f"mov edx, {r32(rn)}",
             f"mul rdx   ; rax *= rdx (unsigned), resultado 64-bit em rax{note}")
        return lines
    mo = re.match(r"^MULS\.W R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"movsx eax, {r16(rm)}", f"movsx edx, {r16(rn)}",
             f"imul eax, edx   ; MACL = produto 16x16 assinado{note}")
        return lines
    mo = re.match(r"^MULU\.W R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"movzx eax, {r16(rm)}", f"movzx edx, {r16(rn)}",
             f"imul eax, edx   ; MACL = produto 16x16 sem sinal{note}")
        return lines
    if m in ("STS MACH,R0", "STS MACH,R1", "STS MACH,R2", "STS MACH,R3", "STS MACH,R4",
             "STS MACH,R5", "STS MACH,R6", "STS MACH,R7") or m.startswith("STS MACH,R"):
        rn = int(re.search(r"R(\d+)$", m).group(1))
        emit(f"mov {r32(rn)}, edx   ; MACH ~ rdx (metade alta do último resultado MAC){note}")
        tr.invalidate(rn)
        return lines
    if m.startswith("STS MACL,R"):
        rn = int(re.search(r"R(\d+)$", m).group(1))
        emit(f"mov {r32(rn)}, eax   ; MACL ~ eax (metade baixa){note}")
        tr.invalidate(rn)
        return lines
    if m.startswith("LDS R") and m.endswith(",MACH"):
        rn = int(re.search(r"R(\d+)", m).group(1))
        emit(f"mov edx, {r32(rn)}{note}")
        return lines
    if m.startswith("LDS R") and m.endswith(",MACL"):
        rn = int(re.search(r"R(\d+)", m).group(1))
        emit(f"mov eax, {r32(rn)}{note}")
        return lines
    if m == "DIV0U":
        emit(f"; DIV0U: zera o estado de divisão (Q/M/T) - sem efeito colateral direto em registrador{note}")
        return lines
    mo = re.match(r"^DIV0S R(\d+),R(\d+)$", m)
    if mo:
        emit(f"; DIV0S: prepara Q=sinal(Rn), M=sinal(Rm), T=Q^M — setup do laço de DIV1 abaixo{note}")
        return lines
    mo = re.match(r"^DIV1 R(\d+),R(\d+)$", m)
    if mo:
        emit(f"; DIV1: 1 passo do algoritmo de divisão bit-a-bit do SH-2 — em x86 isso tudo vira um único "
             f"\"div\"/\"idiv\" (ver bloco DIV0U/DIV0S+DIV1 completo no comentário de interpretação){note}")
        return lines

    # --- extends / sign ---
    mo = re.match(r"^EXTU\.([BW]) R(\d+),R(\d+)$", m)
    if mo:
        width, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        src = r8(rm) if width == "B" else r16(rm)
        emit(f"movzx {r32(rn)}, {src}{note}")
        cur = tr.get_const(rm)
        if cur is not None:
            tr.set_const(rn, cur & (0xFF if width == "B" else 0xFFFF))
        else:
            tr.invalidate(rn)
        return lines
    mo = re.match(r"^EXTS\.([BW]) R(\d+),R(\d+)$", m)
    if mo:
        width, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        src = r8(rm) if width == "B" else r16(rm)
        emit(f"movsx {r32(rn)}, {src}{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^SWAP\.B R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        if rm != rn:
            emit(f"mov {r32(rn)}, {r32(rm)}")
        emit(f"rol {r16(rn)}, 8   ; troca os 2 bytes baixos{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^SWAP\.W R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        if rm != rn:
            emit(f"mov {r32(rn)}, {r32(rm)}")
        emit(f"rol {r32(rn)}, 16{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^MOVT R(\d+)$", m)
    if mo:
        rn = int(mo.group(1))
        cc_true, _ = getattr(tr, "pending_cc", (None, None))
        if cc_true:
            emit(f"set{cc_true} {r8(rn)}   ; materializa o T bit pendente (ver CMP/TST acima)",
                 f"movzx {r32(rn)}, {r8(rn)}{note}")
        else:
            emit(f"; sem CMP/TST rastreado imediatamente antes — condição do T bit não modelada aqui",
                 f"set?? {r8(rn)}", f"movzx {r32(rn)}, {r8(rn)}{note}")
        tr.invalidate(rn)
        return lines

    # --- control transfer (kind handled by caller for delay-slot ordering) ---
    if m == "RTS":
        emit(f"ret{note}")
        return lines
    if m == "RTE":
        emit(f"iret   ; RTE — não modelado em detalhe (fora do escopo de leaf functions de usuário){note}")
        return lines
    if m == "NOP":
        emit(f"nop{note}")
        return lines
    if m == "CLRT":
        emit(f"clc   ; convenção: T ~ CF quando vem de ADDC/SUBC/ROTC{note}")
        return lines
    if m == "SETT":
        emit(f"stc{note}")
        return lines
    if m == "CLRMAC":
        emit(f"xor eax, eax", f"xor edx, edx   ; MACL:MACH = 0{note}")
        return lines
    mo = re.match(r"^JSR @R(\d+)$", m)
    if mo:
        rm = int(mo.group(1))
        emit(f"call {r64(rm)}{note}")
        return lines
    mo = re.match(r"^JMP @R(\d+)$", m)
    if mo:
        rm = int(mo.group(1))
        emit(f"jmp {r64(rm)}{note}")
        return lines
    mo = re.match(r"^BSR 0x([0-9A-F]+)$", m)
    if mo:
        emit(f"call 0x{mo.group(1)}{note}")
        return lines
    mo = re.match(r"^BRA 0x([0-9A-F]+)$", m)
    if mo:
        emit(f"jmp 0x{mo.group(1)}{note}")
        return lines
    mo = re.match(r"^BRAF R(\d+)$", m)
    if mo:
        emit(f"; BRAF: alvo = PC + {r64(int(mo.group(1)))} + 4 (resolvido em tempo de execução){note}",
             f"jmp rax   ; (rax = alvo calculado antes deste ponto)")
        return lines
    mo = re.match(r"^BSRF R(\d+)$", m)
    if mo:
        emit(f"; BSRF: alvo = PC + {r64(int(mo.group(1)))} + 4{note}", f"call rax")
        return lines
    mo = re.match(r"^(BT|BF)(/S)? 0x([0-9A-F]+)$", m)
    if mo:
        kind, target = mo.group(1), mo.group(3)
        cc_true, cc_false = getattr(tr, "pending_cc", (None, None))
        cc = cc_true if kind == "BT" else cc_false
        if cc:
            emit(f"j{cc} 0x{target}{note}")
        else:
            emit(f"; sem CMP/TST rastreado imediatamente antes — condição do T bit não modelada aqui",
                 f"j?? 0x{target}   ; {kind}{note}")
        return lines

    emit(f"; TODO: sem mapeamento x86 automático ainda para \"{m}\"{note}")
    return lines


def translate_function(decs):
    """Returns a list of (sh2_addr, sh2_mnemonic, [x86_lines]) triples, with
    delay-slot instructions re-ordered to execute BEFORE their branch's
    control-transfer effect (real SH-2 semantics), not in their textual
    (post-branch) listing position."""
    tr = Translator()
    tr.pending_cc = (None, None)
    out = []
    i = 0
    DELAY_SLOT_KINDS = ("RTS", "RTE", "BRA", "BSR", "BRAF", "BSRF")
    while i < len(decs):
        dec = decs[i]
        has_delay_slot = (
            dec.mnemonic in DELAY_SLOT_KINDS
            or dec.mnemonic.startswith("JSR")
            or dec.mnemonic.startswith("JMP")
            or re.match(r"^(BT|BF)/S", dec.mnemonic)
        )
        if has_delay_slot and i + 1 < len(decs):
            slot = decs[i + 1]
            slot_lines = translate_one(slot, tr, is_delay_slot=True)
            out.append((slot.addr, slot.mnemonic, slot_lines))
            main_lines = translate_one(dec, tr, is_delay_slot=False)
            out.append((dec.addr, dec.mnemonic, main_lines))
            i += 2
        else:
            lines = translate_one(dec, tr, is_delay_slot=False)
            out.append((dec.addr, dec.mnemonic, lines))
            i += 1
    return out
