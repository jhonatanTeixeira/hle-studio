#!/usr/bin/env python3
"""Mechanical SH-2 -> "Rust-y" pseudocode transliterator - a second emission
backend sitting next to tools/sh2_to_x86.py, not a competing reimplementation
of it.

Why a separate file instead of editing sh2_to_x86.py in place: the actual
per-opcode logic (which mnemonic string means what, how PC-relative pools
resolve, when a register holds a known hardware address via
Translator.hw_target()) is shared, proven code - re-deriving it here would be
exactly the kind of copy-pasted-and-silently-diverging duplicate logic
CLAUDE.md's SOLID section calls out by name (Sh2Decoder vs
Sh2Cpu::execute_instruction). So this module imports sh2_to_x86's `Translator`
and regex/mnemonic-parsing conventions directly and only replaces the LAST
step - what string gets emitted for a given parsed instruction - a Template
Method split (decode -> track-const -> emit), same pattern already named in
CLAUDE.md for cfg.rs's block walk.

This is NOT valid, compilable Rust and is not meant to be. It's the
`tools/sh2_to_x86.py` idea applied to a different, more legible-to-a-porting-
agent output shape: no register aliasing tables to hold in your head, no x86
flag semantics (`adc`/`sbb`/`seto`) - straight-line pseudocode with SH-2's own
register numbers (r0..r15), an explicit `sr_t` boolean for the T flag, and
`goto L<addr>` for branches (this project's basic blocks don't get
reconstructed into real Rust control flow here - that's cfg.rs's job once a
function's shape is actually understood; this stays mechanical, matching
GEMINI.md's "don't guess" rule the same way sh2_to_x86.py already does).

Usage (same input shape sh2dis.decode() already produces):
    python3 tools/sh2_to_rust_pseudo.py <hex_pc> <hex_opcode> [<hex_pc> <hex_opcode> ...]
    # e.g. addresses copied straight out of a traces/calls/<ADDR>_study.md
    # disassembly block, for a quick "what would this look like as
    # pseudocode" spot-check without opening the Rust proposal.
"""
import re
import sys

from .sh2dis import decode  # same decoder study_frame.py/sh2_to_x86.py already trust (vendored into hle-studio)
from .sh2_to_x86 import Translator, region_for, s8  # noqa: F401 - reuse const-tracking + region lookup, don't refork it


def r(n: int) -> str:
    return f"r{n}"


def translate_one_rust(dec, tr: Translator, is_delay_slot: bool = False) -> list[str]:
    """Rust-pseudo equivalent of sh2_to_x86.translate_one - same dispatch
    order and same regexes (copied, not reinvented) so a mnemonic that's
    handled there is handled here too; only the emitted string differs."""
    m = dec.mnemonic
    lines: list[str] = []
    note = "  // (delay slot)" if is_delay_slot else ""

    def emit(*stmts):
        lines.extend(stmts)

    # --- literal pool loads (already resolved to a static constant) ---
    if dec.pool_addr is not None and m.startswith("MOV.L @(0x"):
        n = int(re.search(r",R(\d+)$", m).group(1))
        val = dec.pool_val
        if val is not None:
            emit(f"{r(n)} = 0x{val:08X};{note}")
            tr.set_const(n, val)
        else:
            emit(f"// pool não resolvido em 0x{dec.pool_addr:08X}{note}")
        return lines
    if dec.pool_addr is not None and m.startswith("MOV.W @(0x") and "PC)" in m:
        n = int(re.search(r",R(\d+)$", m).group(1))
        val = dec.pool_val
        if val is not None:
            sval = (val - 0x10000) if val & 0x8000 else val
            emit(f"{r(n)} = 0x{sval & 0xFFFFFFFF:08X}; // sign-extend de word 0x{val:04X}{note}")
            tr.set_const(n, sval & 0xFFFFFFFF)
        else:
            emit(f"// pool não resolvido{note}")
        return lines
    if dec.pool_addr is not None and m.startswith("MOVA"):
        emit(f"{r(0)} = 0x{dec.pool_addr:08X}; // endereço do pool, não seu conteúdo{note}")
        tr.set_const(0, dec.pool_addr)
        return lines

    # --- simple no-operand instructions ---
    SIMPLE = {
        "NOP": "// nop",
        "RTS": "return; // PR ainda não resolvido estaticamente" + note,
        "RTE": "return_from_exception();" + note,
        "CLRT": "sr_t = false;" + note,
        "SETT": "sr_t = true;" + note,
        "CLRMAC": "mach = 0; macl = 0;" + note,
        "DIV0U": "sr_q = false; sr_m = false; sr_t = false;" + note,
    }
    if m in SIMPLE:
        emit(SIMPLE[m])
        return lines

    # --- STC/STS (status/control register reads) ---
    mo = re.match(r"^STC (SR|GBR|VBR),R(\d+)$", m)
    if mo:
        n = int(mo.group(2))
        emit(f"{r(n)} = {mo.group(1).lower()};{note}")
        tr.invalidate(n)
        return lines
    mo = re.match(r"^STS (MACH|MACL|PR),R(\d+)$", m)
    if mo:
        n = int(mo.group(2))
        emit(f"{r(n)} = {mo.group(1).lower()};{note}")
        tr.invalidate(n)
        return lines
    mo = re.match(r"^LDS R(\d+),(PR|MACH|MACL)$", m)
    if mo:
        n = int(mo.group(1))
        emit(f"{mo.group(2).lower()} = {r(n)};{note}")
        return lines
    mo = re.match(r"^LDC R(\d+),(SR|GBR|VBR)$", m)
    if mo:
        n = int(mo.group(1))
        emit(f"{mo.group(2).lower()} = {r(n)};{note}")
        return lines
    mo = re.match(r"^MOVT R(\d+)$", m)
    if mo:
        n = int(mo.group(1))
        emit(f"{r(n)} = sr_t as u32;{note}")
        tr.invalidate(n)
        return lines

    # --- register moves / immediates ---
    mo = re.match(r"^MOV R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"{r(rn)} = {r(rm)};{note}")
        tr.set_const(rn, tr.get_const(rm))
        return lines
    mo = re.match(r"^MOV #(-?\d+),R(\d+)$", m)
    if mo:
        imm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"{r(rn)} = {imm}i32 as u32;{note}")
        tr.set_const(rn, imm & 0xFFFFFFFF)
        return lines

    # --- R15 stack push/pop (R15 IS the SH-2 SP, same as sh2_to_x86.py) ---
    mo = re.match(r"^MOV\.L R(\d+),@-R15$", m)
    if mo:
        rn = int(mo.group(1))
        emit(f"push_u32({r(rn)}); // R15 == sp{note}")
        return lines
    mo = re.match(r"^MOV\.L @R15\+,R(\d+)$", m)
    if mo:
        rn = int(mo.group(1))
        emit(f"{r(rn)} = pop_u32();{note}")
        tr.invalidate(rn)
        return lines
    if m == "STS.L MACL,@-R15":
        emit(f"push_u32(macl);{note}")
        return lines
    if m == "LDS.L @R15+,MACL":
        emit(f"macl = pop_u32();{note}")
        return lines
    mo = re.match(r"^MOV\.L R(\d+),@-R(\d+)$", m)
    if mo:
        rn, rm = int(mo.group(1)), int(mo.group(2))
        emit(f"{r(rm)} -= 4; mem.write_u32({r(rm)}, {r(rn)});{note}")
        return lines
    mo = re.match(r"^MOV\.L @R(\d+)\+,R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"{r(rn)} = mem.read_u32({r(rm)}); {r(rm)} += 4;{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^MOV\.([BW]) R(\d+),@-R(\d+)$", m)
    if mo:
        width, rn, rm = mo.group(1), int(mo.group(2)), int(mo.group(3))
        step = 1 if width == "B" else 2
        emit(f"{r(rm)} -= {step}; mem.write_{ {'B':'u8','W':'u16'}[width] }({r(rm)}, {r(rn)} as { {'B':'u8','W':'u16'}[width] });{note}")
        return lines
    mo = re.match(r"^MOV\.([BW]) @R(\d+)\+,R(\d+)$", m)
    if mo:
        width, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        step = 1 if width == "B" else 2
        ty = {"B": "u8", "W": "u16"}[width]
        emit(f"{r(rn)} = mem.read_{ty}({r(rm)}) as u32; {r(rm)} += {step};{note}")
        tr.invalidate(rn)
        return lines

    # --- register-indirect load/store @Rm, with hardware-register recognition ---
    WIDTH_TY = {"B": "u8", "W": "u16", "L": "u32"}
    mo = re.match(r"^MOV\.([BWL]) @R(\d+),R(\d+)$", m)
    if mo:
        width, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        region, helper = tr.hw_target(rm)
        if helper:
            emit(f"{r(rn)} = {helper}(); // {r(rm)} == 0x{tr.get_const(rm):08X}, registrador conhecido{note}")
        elif region:
            emit(f"// ATENÇÃO: {r(rm)} aponta para \"{region}\" mas o registrador exato ainda não foi "
                 f"identificado - ver docs/memory_map_plan.md",
                 f"{r(rn)} = read_u{ {'B':8,'W':16,'L':32}[width] }_{region.split()[0].lower()}(); // placeholder{note}")
        else:
            cast = "" if width == "L" else f" as u32 /* sign/zero-extend conforme MOV.{width} */"
            emit(f"{r(rn)} = mem.read_{WIDTH_TY[width]}({r(rm)}){cast};{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^MOV\.([BWL]) R(\d+),@R(\d+)$", m)
    if mo:
        width, rn, rm = mo.group(1), int(mo.group(2)), int(mo.group(3))
        region, helper = tr.hw_target(rm)
        if helper or region:
            target = helper or f"write_u{ {'B':8,'W':16,'L':32}[width] }_{region.split()[0].lower()}"
            emit(f"{target}({r(rn)});{note}")
        else:
            emit(f"mem.write_{WIDTH_TY[width]}({r(rm)}, {r(rn)} as {WIDTH_TY[width]});{note}")
        return lines

    # --- indexed by R0: @(R0,Rm) ---
    mo = re.match(r"^MOV\.([BWL]) @\(R0,R(\d+)\),R(\d+)$", m)
    if mo:
        width, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        cast = "" if width == "L" else " as u32"
        emit(f"{r(rn)} = mem.read_{WIDTH_TY[width]}({r(0)} + {r(rm)}){cast};{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^MOV\.([BWL]) R(\d+),@\(R0,R(\d+)\)$", m)
    if mo:
        width, rn, rm = mo.group(1), int(mo.group(2)), int(mo.group(3))
        emit(f"mem.write_{WIDTH_TY[width]}({r(0)} + {r(rm)}, {r(rn)} as {WIDTH_TY[width]});{note}")
        return lines

    # --- displacement forms @(disp,Rm) / @(disp*4,Rm), R0 or Rn ---
    mo = re.match(r"^MOV\.([BW]) @\((\d+)(?:\*2)?,R(\d+)\),R0$", m)
    if mo:
        width, disp, rm = mo.group(1), int(mo.group(2)), int(mo.group(3))
        off = disp * (2 if width == "W" else 1)
        emit(f"{r(0)} = mem.read_{WIDTH_TY[width]}({r(rm)} + {off}) as u32;{note}")
        tr.invalidate(0)
        return lines
    mo = re.match(r"^MOV\.([BW]) R0,@\((\d+)(?:\*2)?,R(\d+)\)$", m)
    if mo:
        width, disp, rm = mo.group(1), int(mo.group(2)), int(mo.group(3))
        off = disp * (2 if width == "W" else 1)
        emit(f"mem.write_{WIDTH_TY[width]}({r(rm)} + {off}, {r(0)} as {WIDTH_TY[width]});{note}")
        return lines
    mo = re.match(r"^MOV\.L @\((\d+)\*4,R(\d+)\),R(\d+)$", m)
    if mo:
        disp, rm, rn = int(mo.group(1)), int(mo.group(2)), int(mo.group(3))
        emit(f"{r(rn)} = mem.read_u32({r(rm)} + {disp * 4});{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^MOV\.L R(\d+),@\((\d+)\*4,R(\d+)\)$", m)
    if mo:
        rn, disp, rm = int(mo.group(1)), int(mo.group(2)), int(mo.group(3))
        emit(f"mem.write_u32({r(rm)} + {disp * 4}, {r(rn)});{note}")
        return lines

    # --- GBR-relative ---
    mo = re.match(r"^MOV\.([BWL]) R0,@\(0x([0-9A-Fa-f]+)(?:\*\d)?,GBR\)$", m)
    if mo:
        width = mo.group(1)
        emit(f"mem.write_{WIDTH_TY[width]}(gbr + 0x{mo.group(2)}, {r(0)} as {WIDTH_TY[width]});{note}")
        return lines
    mo = re.match(r"^MOV\.([BWL]) @\(0x([0-9A-Fa-f]+)(?:\*\d)?,GBR\),R0$", m)
    if mo:
        width = mo.group(1)
        cast = "" if width == "L" else " as u32"
        emit(f"{r(0)} = mem.read_{WIDTH_TY[width]}(gbr + 0x{mo.group(2)}){cast};{note}")
        tr.invalidate(0)
        return lines

    # --- arithmetic (2-operand) ---
    ARITH2 = {"ADD": "+=", "SUB": "-=", "AND": "&=", "OR": "|=", "XOR": "^="}
    for op, rustop in ARITH2.items():
        mo = re.match(rf"^{op} R(\d+),R(\d+)$", m)
        if mo:
            rm, rn = int(mo.group(1)), int(mo.group(2))
            emit(f"{r(rn)} {rustop} {r(rm)};{note}")
            if op in ("AND", "OR", "XOR"):
                tr.invalidate(rn)
            else:
                a, b = tr.get_const(rm), tr.get_const(rn)
                tr.set_const(rn, ((a + b) if op == "ADD" else (b - a)) & 0xFFFFFFFF
                             if (a is not None and b is not None) else None)
            return lines
    mo = re.match(r"^ADD #(-?\d+),R(\d+)$", m)
    if mo:
        imm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"{r(rn)} = {r(rn)}.wrapping_add({imm}i32 as u32);{note}")
        cur = tr.get_const(rn)
        tr.set_const(rn, (cur + imm) & 0xFFFFFFFF if cur is not None else None)
        return lines
    mo = re.match(r"^ADDC R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"let (v, c) = {r(rn)}.overflowing_add({r(rm)}.wrapping_add(sr_t as u32));",
             f"{r(rn)} = v; sr_t = c;{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^SUBC R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"let (v, b) = {r(rn)}.overflowing_sub({r(rm)}.wrapping_add(sr_t as u32));",
             f"{r(rn)} = v; sr_t = b;{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^(ADDV|SUBV) R(\d+),R(\d+)$", m)
    if mo:
        op, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        wrapop = "overflowing_add" if op == "ADDV" else "overflowing_sub"
        emit(f"let (v, ov) = ({r(rn)} as i32).{wrapop}({r(rm)} as i32);",
             f"{r(rn)} = v as u32; sr_t = ov;{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^NEG R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"{r(rn)} = 0u32.wrapping_sub({r(rm)});{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^NEGC R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"let (v, b) = 0u32.overflowing_sub({r(rm)}.wrapping_add(sr_t as u32));",
             f"{r(rn)} = v; sr_t = b;{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^NOT R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"{r(rn)} = !{r(rm)};{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^DIV1 R(\d+),R(\d+)$", m)
    if mo:
        emit(f"// DIV1: um passo de divisão sem-sinal bit-a-bit (SR.Q/SR.M/T) - ver sh2int.c, "
             f"não há tradução direta de 1 linha{note}")
        tr.invalidate(int(mo.group(2)))
        return lines
    mo = re.match(r"^(MULU|MULS)\.W R(\d+),R(\d+)$", m)
    if mo:
        signed = mo.group(1) == "MULS"
        rm, rn = int(mo.group(2)), int(mo.group(3))
        ty = "i16" if signed else "u16"
        emit(f"macl = ({r(rm)} as {ty} as i32 * {r(rn)} as {ty} as i32) as u32;{note}")
        return lines
    mo = re.match(r"^DMUL([US])\.L R(\d+),R(\d+)$", m)
    if mo:
        signed = mo.group(1) == "S"
        ty = "i64" if signed else "u64"
        rm, rn = int(mo.group(2)), int(mo.group(3))
        emit(f"let prod = ({r(rm)} as {ty}).wrapping_mul({r(rn)} as {ty});",
             f"mach = (prod >> 32) as u32; macl = prod as u32;{note}")
        return lines
    mo = re.match(r"^MAC\.W @R(\d+)\+,@R(\d+)\+$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"macl = macl.wrapping_add((mem.read_i16({r(rm)}) as i32 * mem.read_i16({r(rn)}) as i32) as u32); "
             f"{r(rm)} += 2; {r(rn)} += 2;{note}")
        return lines

    # --- comparisons: all set sr_t, none write a register ---
    CMP = {"CMP/EQ": "==", "CMP/HS": ">=u", "CMP/GE": ">=s", "CMP/HI": ">u", "CMP/GT": ">s"}
    mo = re.match(r"^(CMP/EQ|CMP/HS|CMP/GE|CMP/HI|CMP/GT) R(\d+),R(\d+)$", m)
    if mo:
        op, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        if op.endswith(("HS", "HI")):  # unsigned compares
            emit(f"sr_t = {r(rn)} {'>' if op == 'CMP/HI' else '>='} {r(rm)};{note}")
        elif op in ("CMP/GE", "CMP/GT"):
            emit(f"sr_t = ({r(rn)} as i32) {'>' if op == 'CMP/GT' else '>='} ({r(rm)} as i32);{note}")
        else:
            emit(f"sr_t = {r(rn)} == {r(rm)};{note}")
        return lines
    mo = re.match(r"^CMP/EQ #(-?\d+),R0$", m)
    if mo:
        emit(f"sr_t = {r(0)} == {int(mo.group(1))}i32 as u32;{note}")
        return lines
    mo = re.match(r"^CMP/STR R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"sr_t = {r(rm)}.to_be_bytes().iter().zip({r(rn)}.to_be_bytes()).any(|(a,b)| a == b);{note}")
        return lines
    mo = re.match(r"^CMP/P([LZ]) R(\d+)$", m)
    if mo:
        cond, rn = mo.group(1), int(mo.group(2))
        op = ">" if cond == "L" else ">="
        emit(f"sr_t = ({r(rn)} as i32) {op} 0;{note}")
        return lines
    mo = re.match(r"^TST R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"sr_t = ({r(rn)} & {r(rm)}) == 0;{note}")
        return lines
    mo = re.match(r"^TST #0x([0-9A-Fa-f]+),R0$", m)
    if mo:
        emit(f"sr_t = ({r(0)} & 0x{mo.group(1)}) == 0;{note}")
        return lines
    mo = re.match(r"^(AND|OR|XOR) #0x([0-9A-Fa-f]+),R0$", m)
    if mo:
        rustop = {"AND": "&=", "OR": "|=", "XOR": "^="}[mo.group(1)]
        emit(f"{r(0)} {rustop} 0x{mo.group(2)};{note}")
        tr.invalidate(0)
        return lines

    # --- shifts/rotates (all single-register, in place) ---
    mo = re.match(r"^SH([AL])([LR])(\d*) R(\d+)$", m)
    if mo:
        signed, direction, amount, rn = mo.group(1) == "A", mo.group(2), mo.group(3), int(mo.group(4))
        n = int(amount) if amount else 1
        if direction == "L":
            emit(f"sr_t = ({r(rn)} >> 31) != 0; {r(rn)} <<= {n};{note}" if n == 1
                 else f"{r(rn)} <<= {n};{note}")
        else:
            ty = "i32" if signed else "u32"
            emit(f"sr_t = ({r(rn)} & 1) != 0; {r(rn)} = (({r(rn)} as {ty}) >> {n}) as u32;{note}" if n == 1
                 else f"{r(rn)} = (({r(rn)} as {ty}) >> {n}) as u32;{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^ROT([LR])(C?) R(\d+)$", m)
    if mo:
        direction, carry, rn = mo.group(1), mo.group(2) == "C", int(mo.group(3))
        if direction == "L":
            if carry:
                emit(f"let carry_in = sr_t as u32; sr_t = ({r(rn)} >> 31) != 0; "
                     f"{r(rn)} = ({r(rn)} << 1) | carry_in;{note}")
            else:
                emit(f"sr_t = ({r(rn)} >> 31) != 0; {r(rn)} = {r(rn)}.rotate_left(1);{note}")
        else:
            if carry:
                emit(f"let carry_in = (sr_t as u32) << 31; sr_t = ({r(rn)} & 1) != 0; "
                     f"{r(rn)} = ({r(rn)} >> 1) | carry_in;{note}")
            else:
                emit(f"sr_t = ({r(rn)} & 1) != 0; {r(rn)} = {r(rn)}.rotate_right(1);{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^DT R(\d+)$", m)
    if mo:
        rn = int(mo.group(1))
        emit(f"{r(rn)} = {r(rn)}.wrapping_sub(1); sr_t = {r(rn)} == 0;{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^EXTU\.([BW]) R(\d+),R(\d+)$", m)
    if mo:
        width, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        ty = "u8" if width == "B" else "u16"
        emit(f"{r(rn)} = {r(rm)} as {ty} as u32;{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^EXTS\.([BW]) R(\d+),R(\d+)$", m)
    if mo:
        width, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        ty = "i8" if width == "B" else "i16"
        emit(f"{r(rn)} = {r(rm)} as {ty} as i32 as u32;{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^SWAP\.([BW]) R(\d+),R(\d+)$", m)
    if mo:
        width, rm, rn = mo.group(1), int(mo.group(2)), int(mo.group(3))
        if width == "B":
            emit(f"{r(rn)} = ({r(rm)} & 0xFFFF0000) | (({r(rm)} & 0xFF) << 8) | (({r(rm)} >> 8) & 0xFF);{note}")
        else:
            emit(f"{r(rn)} = {r(rm)}.rotate_left(16);{note}")
        tr.invalidate(rn)
        return lines
    mo = re.match(r"^XTRCT R(\d+),R(\d+)$", m)
    if mo:
        rm, rn = int(mo.group(1)), int(mo.group(2))
        emit(f"{r(rn)} = (({r(rn)} >> 16) & 0xFFFF) | (({r(rm)} << 16) & 0xFFFF0000);{note}")
        tr.invalidate(rn)
        return lines

    # --- control flow ---
    mo = re.match(r"^BRA 0x([0-9A-Fa-f]+)$", m)
    if mo:
        emit(f"goto L{mo.group(1)};{note}")
        return lines
    mo = re.match(r"^BSR 0x([0-9A-Fa-f]+)$", m)
    if mo:
        emit(f"call_L{mo.group(1)}();{note}")
        return lines
    mo = re.match(r"^BRAF R(\d+)$", m)
    if mo:
        n = int(mo.group(1))
        emit(f"goto *({r(n)} as *const ()); // alvo dinâmico, não resolvido estaticamente{note}")
        return lines
    mo = re.match(r"^BSRF R(\d+)$", m)
    if mo:
        n = int(mo.group(1))
        emit(f"call_dynamic({r(n)}); // alvo dinâmico, não resolvido estaticamente{note}")
        return lines
    mo = re.match(r"^(BT|BF)(/S)? 0x([0-9A-Fa-f]+)$", m)
    if mo:
        cond, target = mo.group(1), mo.group(3)
        test = "sr_t" if cond == "BT" else "!sr_t"
        emit(f"if {test} {{ goto L{target}; }}{note}")
        return lines
    mo = re.match(r"^JSR @R(\d+)$", m)
    if mo:
        emit(f"call_dynamic({r(int(mo.group(1)))}); // alvo em registrador{note}")
        return lines
    mo = re.match(r"^JMP @R(\d+)$", m)
    if mo:
        emit(f"goto *({r(int(mo.group(1)))} as *const ()); // alvo em registrador{note}")
        return lines
    if m.startswith("DIV0S"):
        emit(f"// DIV0S: inicializa SR.Q/SR.M pros bits de sinal do dividendo/divisor{note}")
        return lines
    mo = re.match(r"^TRAPA #0x([0-9A-Fa-f]+)$", m)
    if mo:
        emit(f"trapa(0x{mo.group(1)});{note}")
        return lines

    # --- anything not covered above: flagged loudly, not silently defaulted
    # (see CLAUDE.md's hard rule against a blanket `_ => DATA`-style fallback
    # - an unhandled mnemonic here must look unhandled, not like a no-op). ---
    emit(f"// TODO sh2_to_rust_pseudo: mnemônico não traduzido: {m}{note}")
    return lines


def translate_function(decs) -> list[str]:
    """Same per-function driver shape as sh2_to_x86.translate_function:
    one Translator across the whole straight-line instruction list, one
    `L<addr>:` label per instruction so BRA/BSR/BT/BF targets resolve to a
    real anchor even though there's no real block/loop reconstruction here."""
    tr = Translator()
    out = []
    for i, dec in enumerate(decs):
        out.append(f"L{dec.addr:08X}:")
        is_delay = decs[i - 1].mnemonic.split()[0] in ("BRA", "BSR", "BRAF", "BSRF", "JSR", "JMP") if i > 0 else False
        for line in translate_one_rust(dec, tr, is_delay_slot=is_delay):
            out.append(f"    {line}")
    return out


def main():
    if len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print(__doc__)
        sys.exit(1)
    pairs = [(int(sys.argv[i], 16), int(sys.argv[i + 1], 16)) for i in range(1, len(sys.argv), 2)]
    decs = [decode(pc, op) for pc, op in pairs]
    for line in translate_function(decs):
        print(line)


if __name__ == "__main__":
    main()
