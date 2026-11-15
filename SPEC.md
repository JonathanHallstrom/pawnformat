<h1 align="center">
    PawnFormat
</h1>

<h5 align="center">
    Status: Draft — Version: 1
</h5>

## Overview

PawnFormat is a compact binary format for storing chess games—along with a score (evaluation) for every position—
intended as training data for neural networks, with an explicit focus on compression and speed.

Two ideas do most of the work:

- **Moves** are stored as an index into the sorted list of legal moves of the position they are played in. As such, a
  move never costs more than 8 bits—and a forced move costs nothing at all.
- **Scores** are stored as the difference from the previous score, in a variable-width code that adapts to how quiet or
  sharp the game currently is.

| Section | Covers |
|---|:--|
| [1 — Definitions and Conventions](#section-1--definitions-and-conventions) | Notation, squares, bit order, helper methods |
| [2 — Structure](#section-2--structure) | Segments, file header, game layout, moves, scores |
| [3 — Reference Reader](#section-3--reference-reader-informative) | A complete reader in pseudocode (informative) |
| [4 — Conformance](#section-4--conformance) | What a reader rejects |
| [Appendices](#appendix-a--recommended-parameters) | Recommended parameters, sizes, a worked example |

---

## Section 1 — Definitions and Conventions

### 1.1 — Keywords

Keywords such as **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are to be interpreted as
described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

### 1.2 — Notation

| Notation | Meaning |
|---|:--|
| `uN` | Unsigned integer of width `N`, holding a value in `[0, 2^N)` |
| `iN` | Two's-complement signed integer of width `N`, holding a value in `[-2^(N-1), 2^(N-1))` |
| `u(A..B)` | Unsigned integer of width `N`, where `A <= N <= B` and `N` is context-dependent |
| `i(A..B)` | Two's-complement signed integer of width `N`, where `A <= N <= B` and `N` is context-dependent |
| `u(A op B)` | Unsigned integer of width `N := A op B`, for any arithmetic operation `op` |
| `i(A op B)` | Two's-complement signed integer of width `N := A op B`, for any arithmetic operation `op` |
| `T[N]` | `N` consecutive instances of `T`—analogous to a contiguous array of `T` |
| `T X = { C }` | Field `X` of type `T` **MUST** hold the constant `C` which is a valid `T` |
| `eg(k)` | Order-`k` Exp-Golomb code—variable width (see [2.6.4](#264--exp-golomb-code-egk)) |

> [!NOTE]
> A field that can be simplified to either `u0` or `i0` occupies no bits at all. As such, `u(0..8)`—and
> likewise, `i(0..8)`—can vanish entirely when its context permits.

> [!NOTE]
> Outside of pseudocode, `A^B` means `A` raised to the power `B`. Inside pseudocode, `^` is a bitwise XOR instead.

### 1.3 — Squares, Ranks, and Files

Files `[A, ..., H]` and ranks `[1, ..., 8]` are both indexed as `[0, ..., 7]`. As such, a square's index can be
calculated as `8 * r + f`, where `r` is the index of its rank and `f` is the index of its file.

> [!TIP]
> The **back rank** of White is `r = 0`, and that of Black is `r = 7`.

An important interpretation of `u64` in chess is a **bitboard**, where each bit directly corresponds to a square on the
chess board (a sort of mask). Bit `i` of a bitboard corresponds to square `i`:

|   | A | B | C | D | E | F | G | H |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **8** | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 |
| **7** | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 |
| **6** | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 |
| **5** | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 |
| **4** | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
| **3** | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| **2** | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| **1** | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

### 1.4 — Endianness

All data is little-endian, LSB-first:
```py
write_bits(x, n):
    for i in 0 .. n - 1:
        write_bit((x >> i) & 1)

read_bits(n):
    x = 0

    for i in 0 .. n - 1:
        x |= read_bit() << i

    return x
```

> [!TIP]
> Consequently, any byte-aligned `u16`/`u64` is plain little-endian—any `amd64` or `arm64` reader can load it directly.

### 1.5 — Helper Methods

To make it easier for the rest of the document to convey the specification, the following methods are defined:
```py
count_leading_zeros(x):
    n = 0

    for i in 63 .. 0:
        if (x >> i) & 1 == 0:
            n += 1
        else:
            break

    return n

bit_width(x):
    if x == 0:
        return 0
    else:
        return 64 - count_leading_zeros(x)

popcount(x):
    n = 0

    for i in 0 .. 63:
        if (x >> i) & 1 == 1:
            n += 1

    return n

zigzag(x):
    return (x << 1) ^ (x >> 63)

unzigzag(x):
    return (x >> 1) ^ (0 - (x & 1))
```

`zigzag` folds signed values into unsigned ones (and likewise, `unzigzag` unfolds them back):

| `x` | `0` | `-1` | `1` | `-2` | `2` | `-3` | `3` | … |
|---|---|---|---|---|---|---|---|---|
| `zigzag(x)` | `0` | `1` | `2` | `3` | `4` | `5` | `6` | … |

> [!TIP]
> A handy identity used later: for `n >= 1`, `bit_width(n - 1) = ceil(log2(n))`—the number of bits required to
> represent `n` distinct values.

## Section 2 — Structure

A PawnFormat file is a sequence of one or more **segments**. A segment is one file header (`file_header`) followed by
zero or more games (`game`). Grammatically, this can be expressed as:
```
file    := segment+
segment := file_header game*
```

Every file header and every game begins on a byte boundary and occupies a whole number of bytes. A file **MUST** begin
with a file header, and a file **MUST** end exactly at the end of a game or file header.

### 2.1 — Segment Detection

Every game begins with a non-zero `u64` (`occupancy` bitboard), which at minimum contains both kings. Every file header
begins with a zero `u64` (`magic` bytes)—a reader expecting either one distinguishes the two by inspecting the
next 8 bytes:

| Next 8 bytes | Meaning |
|---|---|
| Fewer than 8 remain | Error |
| All zero | A `file_header` follows |
| Anything else | A `game` follows |

### 2.2 — Concatenation of Segments

> [!TIP]
> Concatenating datasets is just `cat a b > c`—even when `a` and `b` were encoded with different parameters.

Thanks to [2.1](#21--segment-detection), the byte-wise concatenation of one or more valid files is a valid file. Each
segment carries its own coding parameters: a reader **MUST** apply the parameters of the most recent file header to
each game.

### 2.3 — File Header (12 bytes)

```
file_header {
    u64 magic = { 0 };    // file header marker

    u16 ema_init;         // initial EMA state v at the start of every game

    u4 ema_old_weight;    // weight on the previous v
    u4 ema_new_weight;    // weight on the latest abs(d)

    u4 ema_shift;         // right shift normalizing the weighted sum

    u4 k_cap;             // maximum exp-golomb order
}
```

These parameters drive the adaptive score coding in [2.6.3](#263--adaptive-order), and their recommended values are
listed in [Appendix A](#appendix-a--recommended-parameters). For brevity, the rest of this document refers to them by
the following symbols:

| Field | Symbol |
|---|:-:|
| `ema_init` | `v_0` |
| `ema_old_weight` | `w_o` |
| `ema_new_weight` | `w_n` |
| `ema_shift` | `s` |
| `k_cap` | `k_max` |

> [!IMPORTANT]
> `w_o + w_n` **MUST** equal `2^s`. Since both weights are at most `15`, this implies `2^s <= 30` and thus, it also
> implies that `s <= 4`. A reader **MUST** reject a file header violating it.

This constraint is what keeps the EMA state `v` bounded. Every new state is a weighted average of the previous state and
the latest `abs(d)`—and an average can never exceed the larger of the two values going into it. Since both of them
always fit in 16 bits, so does the result.

> [!TIP]
> Consequently, `v` always fits in a `u16`, and every intermediate product stays below `2^20`—a `u32` is more than
> enough to compute it.

> [!NOTE]
> Degenerate settings are permitted. Weights `(w_o, w_n) = (1, 0)` with `s = 0` never change `v` (giving a constant
> `k`), while weights `(0, 1)` with `s = 0` derive `k` from the previous delta alone.

### 2.4 — Game

```
game {
    game_header {
        u64 occupancy;

        u(8..128) piece_types;

        u4 en_passant_file;

        u1 side_to_move;
        u7 halfmove_clk;

        u2 result;

        u16 ply_count;
    }

    ply_data[ply_count] {
        u(0..8) move_index_in_list;    // see 2.5

        eg(k) {
            u(z) zeros = { 0 };

            u1 one = { 1 };

            u(z + k) suffix;
        } score_delta;                 // see 2.6
    }

    u(0..7) padding = { 0 };
}
```

The fields are written back-to-back in the order shown, with no gaps other than `padding`. Only the game as a whole is
byte-aligned—the fields inside it can (and do) straddle byte boundaries.

The game header (`game_header`) describes the position the game starts from, which is not necessarily the usual starting
position of chess. Its fields are defined below, while the two fields of `ply_data` are defined in [2.5](#25--moves) and
[2.6](#26--scores).

#### 2.4.1 — `occupancy`

A bitboard of every occupied square, regardless of which piece is standing on it. With `p` being
`popcount(occupancy)`—the number of pieces on the board—a reader **MUST** reject a game where `p < 2` or `p > 32`.

> [!TIP]
> Both kings are always on the board, so `occupancy` is never zero. This is exactly what makes
> [2.1](#21--segment-detection) work.

#### 2.4.2 — `piece_types`

One 4-bit piece code per occupied square, in ascending square order. As such, the field is `4 * p` bits wide—anywhere
from 8 to 128 bits:
```py
read_pieces(occupancy):
    for square in 0 .. 63:
        if (occupancy >> square) & 1 == 1:
            code = read_bits(4)

            type  = code & 7     # bits 0..2
            color = code >> 3    # bit 3

            place(square, type, color)
```

| Color | Side |
|:-:|---|
| `0` | White |
| `1` | Black |

| Type | Piece |
|:-:|---|
| `0` | Pawn |
| `1` | Knight |
| `2` | Bishop |
| `3` | Rook |
| `4` | Queen |
| `5` | King |
| `6` | Castling Rook |
| `7` | *Reserved* |

A **castling rook** is a rook with which its side still has the right to castle. Other than that, it is a rook: every
rule that applies to rooks applies to it too. A castling rook is on the **A-side** if its file is lower than that of its
king, and on the **H-side** if its file is higher.

> [!NOTE]
> Castling rights live on the rooks themselves, rather than in a separate field. This handles standard chess and
> Chess960 alike, with no ambiguity about *which* rook can castle.

A reader **MUST** reject a game header whose position contains any of the following:

| Rule | Reject when |
|---|---|
| Reserved type | Any piece is of type `7` |
| Kings | A side has anything other than exactly one king |
| Pawns | A pawn is on `r = 0` or `r = 7` |
| Castling rook placement | A castling rook is not on its side's back rank, or its king is not on that back rank |
| Castling rook count | A side has more than one castling rook on the same side (A or H) of its king |
| Check | The side **not** to move is in check (this includes kings standing next to each other) |

> [!NOTE]
> The last rule depends on `side_to_move`, which comes *after* `piece_types`. As such, a position can only be validated
> once the whole game header is read.

Nothing else about the position is checked: any amount of material is fine, the king and castling rooks may be on any
files, and an ordinary rook **MAY** stand further out than a castling rook on the same side. Sections [2.5](#25--moves)
and [2.6](#26--scores) apply to all such positions as written.

#### 2.4.3 — `en_passant_file`

| Value | Meaning |
|---|---|
| `0..7` | File `f` of the en passant target square |
| `8` | No en passant target |
| `9..15` | *Reserved*—a reader **MUST** reject them |

An en passant target exists right after a pawn advances two squares, when an opposing pawn is able to capture it as if
it had only advanced one. Three squares are involved, all of them on file `f`:

| Square | Meaning | White to move | Black to move |
|---|---|:-:|:-:|
| Target `t` | Where the capturing pawn lands | `40 + f` | `16 + f` |
| Advanced pawn `c` | The pawn that just advanced two squares | `32 + f` | `24 + f` |
| Origin `o` | Where that pawn advanced from | `48 + f` | `8 + f` |

An **en passant capture** moves a pawn of the side to move—standing right next to `c`, on file `f - 1` or `f + 1`—to
`t`, and removes the pawn on `c` from the board.

> [!TIP]
> For example, say Black has just advanced a pawn from D7 to D5, and White is to move with `en_passant_file = 3`. Then
> `t` is D6 (square 43), `c` is D5 (square 35), and `o` is D7 (square 51). A White pawn on C5 or E5 can capture by
> moving to D6, which removes the Black pawn on D5.

The target is **valid** if and only if all of the following hold:

1. `t` and `o` are both empty.
2. `c` holds a pawn of the side not to move.
3. At least one en passant capture is legal.

A reader **MUST** reject an invalid target.

> [!CAUTION]
> "Legal" here means fully legal ([2.5.4](#254--move-rules)). A pawn merely standing next to `c` is not enough: if
> capturing would leave its own king attacked (the pawn is pinned, or removing both pawns opens up a line onto the
> king), then that capture doesn't count.

> [!IMPORTANT]
> `en_passant_file` **MUST** be `f` if and only if the previous move was a two-square advance of the pawn now on `c`
> *and* the target is valid. Otherwise, it **MUST** be `8`. This makes the encoding of every position unique, so
> identical positions always produce identical bytes.

A reader can't see the move before the game header, so it only checks that the target is valid—getting the "previous
move" part right for the game header is on the writer. For every later position, the reader applies this very same rule
itself ([2.5.4](#254--move-rules)).

#### 2.4.4 — `side_to_move` and `halfmove_clk`

| Field | Meaning |
|---|---|
| `side_to_move` | `0` for White, `1` for Black |
| `halfmove_clk` | Plies since the last capture or pawn move, saturating at `127` |

In other words, if `h` is the actual half-move clock of the position, then the stored value is `min(h, 127)`.

> [!NOTE]
> The clock never affects which moves are legal ([2.5.2](#252--canonical-move-list)). Saturating at `127` keeps the
> 50-move threshold (`h = 100`) exact, but the 75-move threshold (`h = 150`) is not representable.

No consistency between `halfmove_clk`, `en_passant_file`, and `result` is required. For example, an en passant target
means that a pawn has just moved (so the clock ought to be `0`), yet a non-zero clock next to it is perfectly valid.

#### 2.4.5 — `result`

The result of the played game—how it actually ended, rather than anything about the position in the game header. It is
always from White's perspective, no matter which side is to move:

| Value | Meaning |
|:-:|---|
| `0` | Black wins |
| `1` | Draw |
| `2` | White wins |
| `3` | *Reserved*—a reader **MUST** reject it |

#### 2.4.6 — `ply_count`

The number of `ply_data` entries that follow, which can be anywhere from `0` to `65535`. A game where `ply_count` is `0`
is valid: it holds a position and a result, with no moves and no scores.

#### 2.4.7 — `padding`

Zero bits up to the next byte boundary (so anywhere from 0 to 7 of them). A reader **MUST** reject non-zero padding.

#### 2.4.8 — Independence and Splitting

Games are independent: no state carries over from one game to the next.

A played game longer than 65535 plies (or one that a writer simply chooses to split) is stored as consecutive games. The
game header of each game after the first holds the position reached at its split point, and every one of these games
carries the `result` of the played game. Nothing links them together.

### 2.5 — Moves

This section defines `move_index_in_list`, the first field of every `ply_data` entry.

#### 2.5.1 — Positions

Let `P_0` be the position described by the game header, and let `T` be `ply_count`. Each `ply_data[i]` holds the move
`m_i` played in position `P_i`, along with the score of `P_i`. Playing `m_i` in `P_i` leads to `P_(i+1)`:
```
P_0 --m_0--> P_1 --m_1--> P_2 --m_2--> ... --m_(T-1)--> P_T
```

> [!NOTE]
> That is `T` moves and `T` scores, but `T + 1` positions. The final position `P_T` has neither a move nor a
> score—`result` is what describes the outcome.

#### 2.5.2 — Canonical Move List

A move is never stored directly. Instead, both the writer and the reader build the very same list of moves for the
position, and only the index of the played move is stored.

That list, `L_i`, holds every **legal** move in `P_i` ([2.5.4](#254--move-rules)), sorted in ascending order by `from`,
then by `to`, and then by `promotion`:

| Key | Meaning |
|---|---|
| `from` | Square the moving piece starts on—the king's square for castling |
| `to` | Square the moving piece lands on—**the castling rook's square** for castling |
| `promotion` | `0` none, `1` knight, `2` bishop, `3` rook, `4` queen |

No two moves of a position share the same key, so the order is never ambiguous.

> [!NOTE]
> Castling is represented as "king captures own rook". This is unambiguous in both standard chess and Chess960, even
> when the king's landing square is one it could also reach with an ordinary step.

> [!TIP]
> For example, the usual starting position has 20 legal moves. Sorted, they are the two moves of the knight on B1
> (square 1), the two moves of the knight on G1 (square 6), and then two moves for each pawn from A2 (square 8) to H2
> (square 15). As such, E2 to E4 is at index 13.

The list depends only on piece placement, side to move, castling rooks, and the en passant target. `halfmove_clk`,
repetition, and insufficient material never remove moves or end the game—as far as the format is concerned, a game can
carry on straight past a draw by the 50-move rule.

#### 2.5.3 — `move_index_in_list`

With `n_i` being the number of moves in `L_i`:

- `n_i` **MUST** be at least 1 and at most 218.
- The field is `bit_width(n_i - 1)` bits wide, which is `ceil(log2(n_i))`.
- The value **MUST** be less than `n_i`, and `m_i` is the move at that index of `L_i`.

```py
read_move(position):
    moves = canonical_legal_moves(position)
    n     = length(moves)

    if n < 1 or n > 218:
        reject()

    index = read_bits(bit_width(n - 1))

    if index >= n:
        reject()

    return moves[index]
```

The width grows with the number of legal moves, and forced moves cost nothing at all:

| `n_i` | `1` | `2` | `3..4` | `5..8` | `9..16` | `17..32` | `33..64` | `65..128` | `129..218` |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Bits** | `0` | `1` | `2` | `3` | `4` | `5` | `6` | `7` | `8` |

> [!NOTE]
> A checkmate or stalemate position has no legal moves (`n_i = 0`), so no move can ever be stored for it. Such a
> position can only ever be the final position `P_T`.

> [!IMPORTANT]
> The format supports only legal chess positions, and 218 is the maximum number of legal moves in such a position. A
> writer **MUST NOT** encode a position with more than 218 legal moves, and a reader **MUST** reject one—this also
> catches unreachable positions that slip through the checks of [2.4.2](#242--piece_types). Since `ceil(log2(218)) = 8`,
> the field never exceeds 8 bits.

> [!NOTE]
> 218 is
> [proven](https://lichess.org/@/Tobs40/blog/there-is-no-reachable-chess-position-with-more-than-218-moves/a5xdxeqs) for
> positions reachable in standard chess. A Chess960 position without a legal castle is covered by the same proof, while
> one with a legal castle could—in theory—exceed it by up to two moves. No such position is known, and the cap keeps the
> field at 8 bits regardless.

#### 2.5.4 — Move Rules

Moves follow the rules of chess, with the following made explicit.

**Legality.** A move is legal if and only if it obeys the movement rules of its piece, and the king of the moving side
is not attacked in the resulting position.

**Promotion.** A pawn reaching `r = 7` (White) or `r = 0` (Black) becomes the chosen piece. Promoting to a rook produces
an ordinary rook (type `3`)—never a castling rook.

**En passant.** As defined in [2.4.3](#243--en_passant_file).

**Castling.** The king castles with one of its side's castling rooks. No matter where the two start from, they always
land on the same files of their back rank:

| Side | King lands on | Castling rook lands on |
|---|:-:|:-:|
| A-side | C-file (`f = 2`) | D-file (`f = 3`) |
| H-side | G-file (`f = 6`) | F-file (`f = 5`) |

The **king's path** is every back-rank square from the king's square to its landing square (both included), and the
**rook's path** is likewise for the castling rook. Castling is legal if and only if all of the following hold:

1. The king is not in check.
2. Every square on either path is either empty, or holds the king or the castling rook themselves.
3. No square on the king's path, other than the one it starts on, is attacked by the opponent with both the king and the
   castling rook removed from the board.
4. The king is not attacked in the resulting position.

Castling is permitted even when the king doesn't move at all (it is already on its landing square), and when the king
and the castling rook simply swap squares.

> [!CAUTION]
> Condition 3 removes the castling rook from the board on purpose. A castling rook can shield its own king from an
> attack along the back rank, and once it moves to the D- or F-file, that shield is gone. Condition 4 is not redundant
> either: when the king doesn't move, condition 3 has no squares to check at all.

> [!TIP]
> In standard chess, this boils down to the following four moves:
>
> | Castle | King | Castling rook | Stored as `(from, to)` |
> |---|:-:|:-:|:-:|
> | White H-side | E1 to G1 | H1 to F1 | `(4, 7)` |
> | White A-side | E1 to C1 | A1 to D1 | `(4, 0)` |
> | Black H-side | E8 to G8 | H8 to F8 | `(60, 63)` |
> | Black A-side | E8 to C8 | A8 to D8 | `(60, 56)` |

**Castling state.** The right to castle is lost as follows:

- When a king moves (castling included), every castling rook of its side becomes an ordinary rook.
- When a castling rook moves, it becomes an ordinary rook. When it is captured, the right is simply gone with it.

**Clocks and state.** After every move, `side_to_move` flips. `halfmove_clk` resets to `0` after a capture or pawn move,
and otherwise becomes `min(h + 1, 127)`. The `en_passant_file` of the new position is set by the rule in
[2.4.3](#243--en_passant_file).

> [!WARNING]
> Castling is **not** a capture, despite being encoded as one. An implementation that resets `halfmove_clk` whenever the
> `to` square is occupied will get this wrong.

### 2.6 — Scores

This section defines `score_delta`, the second field of every `ply_data` entry.

#### 2.6.1 — Semantics

The score of a position is an `i16` evaluation **from White's perspective**—positive favors White, no matter which side
is to move. Its unit (typically centipawns) and how it represents mates are up to whoever produced the scores. Every
`i16` value is permitted.

> [!TIP]
> Scores are White-relative so that consecutive scores stay close to each other. Side-to-move-relative scores would flip
> sign every ply, which would roughly double every delta and waste bits.

#### 2.6.2 — Delta Coding

Scores are not stored directly. Instead, each score is stored as its difference `d` (delta) from the previous score of
the same game, where the score "before" the first one is taken to be `0`.

As such, the first delta of a game is the first score itself, while every later delta is somewhere in `[-65535, 65535]`.

Each delta is then stored as `eg(k)` of `zigzag(d)`: `zigzag` turns the signed delta into a small unsigned value, and
`eg(k)` writes that value in as few bits as it can. Note that `zigzag` treats `d` as an `i64`, so its `>>` is an
arithmetic (sign-extending) shift.

#### 2.6.3 — Adaptive Order

The order `k` controls how `eg(k)` spends its bits: a small `k` is cheap for small values but expensive for large ones,
and a large `k` is the other way around ([2.6.4](#264--exp-golomb-code-egk)).

No single `k` suits a whole game. As such, the format keeps a state `v`—an exponential moving average (EMA) of
`abs(d)`—and derives `k` from it: every game starts with `v` at `v_0`, and each `k` is the `bit_width` of `v`, capped at
`k_max`. Quiet stretches end up with a small `k`, and sharp ones with a large `k`:
```py
start_game():
    prev = 0
    v    = ema_init

write_score(score):
    k = min(bit_width(v), k_cap)
    d = score - prev

    write_eg(zigzag(d), k)

    prev = score
    v    = (ema_old_weight * v + ema_new_weight * abs(d)) >> ema_shift

read_score():
    k = min(bit_width(v), k_cap)
    d = unzigzag(read_eg(k))

    score = prev + d

    if score < -32768 or score > 32767:
        reject()

    prev = score
    v    = (ema_old_weight * v + ema_new_weight * abs(d)) >> ema_shift

    return score
```

> [!IMPORTANT]
> `k` is always computed from `v` **before** `v` is updated with the current delta. The reader doesn't know that delta
> yet, so this is the only way it can reproduce the very same `k` as the writer.

Here's a short walkthrough with the [recommended parameters](#appendix-a--recommended-parameters), where `v` starts at
`64`:

| Ply | Score | `d` | `zigzag(d)` | `k` | Bits | Next `v` |
|:-:|--:|--:|--:|:-:|:-:|--:|
| 0 | 30 | +30 | 60 | 7 | 8 | 51 |
| 1 | 25 | -5 | 9 | 6 | 7 | 33 |
| 2 | 40 | +15 | 30 | 6 | 7 | 26 |
| 3 | 38 | -2 | 3 | 5 | 6 | 17 |
| 4 | -150 | -188 | 375 | 5 | 12 | 81 |
| 5 | -140 | +10 | 20 | 7 | 8 | 54 |

That's 48 bits for six scores. The blunder at ply 4 costs 12 bits instead of something far larger, and `k` immediately
widens to absorb the follow-up.

#### 2.6.4 — Exp-Golomb Code `eg(k)`

An order-`k` Exp-Golomb code stores an unsigned value `u` in three parts:

| Part | Width | Holds |
|---|:-:|---|
| `zeros` | `z` | Zero bits—counting them tells the reader how wide `suffix` is |
| `one` | `1` | A one bit, ending the run of zeros |
| `suffix` | `z + k` | The low `z + k` bits of `u + 2^k` (its highest set bit is implied, so it isn't stored) |

```py
write_eg(u, k):
    x = u + (1 << k)
    z = bit_width(x) - 1 - k

    write_bits(0, z)
    write_bit(1)
    write_bits(x & ((1 << (z + k)) - 1), z + k)

read_eg(k):
    z = 0

    while read_bit() == 0:
        z += 1

        if z + k > 17:
            reject()

    suffix = read_bits(z + k)

    return ((1 << (z + k)) | suffix) - (1 << k)
```

That's `2 * z + k + 1` bits in total. Values are grouped by `z`, and each group is twice as wide as the last:

| `z` | Range of `u` | Bits |
|:-:|---|:-:|
| `0` | `[0, 2^k)` | `k + 1` |
| `1` | `[2^k, 3 * 2^k)` | `k + 3` |
| `2` | `[3 * 2^k, 7 * 2^k)` | `k + 5` |
| `3` | `[7 * 2^k, 15 * 2^k)` | `k + 7` |
| … | … | … |

> [!TIP]
> For example, take `d = 3` with `k = 2`. Then `u` is `zigzag(3) = 6`, and `u + 2^k = 10` (`1010` in binary), so
> `z = 1`. The stream holds the five bits `0 1 0 1 0`: one zero, the one, and then the low three bits of `1010`
> (LSB-first).

> [!TIP]
> With LSB-first bit order, counting the zeros is a single trailing-zero count (`tzcnt`) on a buffered 64-bit word.

Every valid code is bounded. The largest possible `u` is `zigzag(65535) = 131070` and the largest possible `k` is `15`
(since `k_cap` is a `u4`), so `u + 2^k` always stays below `2^18`. As such, `z + k` never exceeds `17`, and no valid
code is longer than 34 bits.

> [!IMPORTANT]
> A reader **MUST** reject a code as soon as its zero count makes `z + k > 17`—rather than after scanning the whole run
> of zeros—so that corrupt data can never cause an unbounded scan. Codes within that bound can still decode to a delta
> that puts the score outside `i16`, and a reader **MUST** reject such a score as well.

## Section 3 — Reference Reader (Informative)

This section puts the rules above together as a reference reader, using the methods defined along the way. The
parameters used by `start_game` and `read_score` are those of the most recent file header:
```py
read_file():
    file_header = none

    while not end_of_file():
        if fewer than 8 bytes remain:
            reject()

        if the next 8 bytes are all zero:
            file_header = read_file_header()    # validates 2.3
        else:
            if file_header == none:
                reject()

            read_game()

read_game():
    position = read_game_header()               # validates 2.4.1 to 2.4.5

    start_game()                                # 2.6.3

    for i in 0 .. position.ply_count - 1:
        move  = read_move(position)             # 2.5.3
        score = read_score()                    # 2.6.3

        emit(position, move, score, position.result)

        position = make_move(position, move)    # 2.5.4

    if any bit up to the next byte boundary is not zero:
        reject()
```

## Section 4 — Conformance

A reader **MUST** treat each of the following as an error:

| # | Error | See |
|:-:|---|:-:|
| 1 | A file not beginning with a file header, or ending in the middle of a file header or game | [2](#section-2--structure) |
| 2 | A file header where `w_o + w_n` does not equal `2^s` | [2.3](#23--file-header-12-bytes) |
| 3 | A game header violating any rule of 2.4.1 to 2.4.5 | [2.4](#24--game) |
| 4 | A ply with no legal moves or more than 218 of them, or whose `move_index_in_list` is not less than `n_i` | [2.5.3](#253--move_index_in_list) |
| 5 | A score code whose zero count makes `z + k > 17`, or a decoded score outside `i16` | [2.6.4](#264--exp-golomb-code-egk) |
| 6 | Non-zero padding | [2.4.7](#247--padding) |

A writer **MUST** produce only data that a conforming reader accepts, and **MUST** follow the en passant rule of
[2.4.3](#243--en_passant_file).

## Appendix A — Recommended Parameters

| Field | Symbol | Value |
|---|:-:|:-:|
| `ema_init` | `v_0` | `64` |
| `ema_old_weight` | `w_o` | `5` |
| `ema_new_weight` | `w_n` | `3` |
| `ema_shift` | `s` | `3` |
| `k_cap` | `k_max` | `10` |

With these, the first score of every game is coded with `k = 7` (the `bit_width` of `64`). Each ply keeps 5/8 of the
previous state, which is fast enough to switch between the quiet and sharp phases of a game within a move or two. A run
of zero deltas drives `v`—and with it, `k`—all the way down to `0`.

## Appendix B — Sizes

| What | Size |
|---|---|
| `file_header` | 96 bits (12 bytes) per segment |
| `game_header` | `94 + 4 * p` bits, so anywhere from 102 to 222 |
| `ply_data[i]` | `ceil(log2(n_i)) + 2 * z_i + k_i + 1` bits |
| `padding` | 0 to 7 bits per game |

Putting it together, a game with `p` pieces and `T` plies occupies exactly:
```
bits  = 94 + 4 * p + (the bits of every ply_data entry)
bytes = ceil(bits / 8)
```

## Appendix C — Worked Example

A complete (and tiny) file: one file header with the [recommended parameters](#appendix-a--recommended-parameters),
followed by one game of a single ply.

The file header is 12 bytes:
```
00 00 00 00 00 00 00 00 40 00 35 A3
```

| Field | Bits | Value |
|---|:-:|---|
| `magic` | 64 | `0` |
| `ema_init` | 16 | `64` |
| `ema_old_weight` | 4 | `5` |
| `ema_new_weight` | 4 | `3` |
| `ema_shift` | 4 | `3` |
| `k_cap` | 4 | `10` |

> [!TIP]
> Notice the byte `35`: `ema_old_weight` is written first, so it lands in the *low* half of the byte. That's LSB-first
> at work ([1.4](#14--endianness)).

The game has the White king on E1 and the Black king on E8, with White to move. White plays the king from E1 to E2 in a
position scored `12`, and the game ends in a draw. It is 15 bytes:
```
10 00 00 00 00 00 00 10 D5 08 50 00 C0 62 00
```

| Field | Bits | Value |
|---|:-:|---|
| `occupancy` | 64 | Squares 4 and 60 |
| `piece_types` | 8 | `5` (White king), then `13` (Black king) |
| `en_passant_file` | 4 | `8` (none) |
| `side_to_move` | 1 | `0` (White) |
| `halfmove_clk` | 7 | `0` |
| `result` | 2 | `1` (draw) |
| `ply_count` | 16 | `1` |
| `move_index_in_list` | 3 | `3`—the king has 5 legal moves (to squares 3, 5, 11, 12, and 13), and E2 is square 12 |
| `score_delta` | 8 | `d = 12`, so `zigzag(d) = 24`, coded with `k = 7` and `z = 0` |
| `padding` | 7 | `0` |

That's 113 bits of data, padded up to 120 bits (15 bytes).
