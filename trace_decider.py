"""Trace one Decider forward pass by hand and check it against the public API.

Rebuilds the prompt rows that `system_one` builds, runs the backbone once, reads the
answer slot, slices the output head to the option letters, and applies the calibration
temperature. Every number the blog quotes for Decider is printed here.

    uv run python trace_decider.py

Sections:
    1  prompt      token count, answer slot, the last four token ids
    2  head        the tied head, the 255-row slice, the letter ids
    3  choice      raw logits, masked logits, softmax at T=1 and at the config T
    4  api         the same question through system_one, key for key
    5  vocabulary  the same hidden state against all 248,320 rows, unmasked
    6  padding     the slot hidden state with and without the pad to a multiple of 64
    7  noul        the two-option row that a Noul answer is read from
    8  score       isolated levels, fit_mass, and the listwise readout for comparison
"""

import os
import sys

import torch
import torch.nn.functional as F

DECIDER_MODEL = "Mapika/decider-4b"
DECIDER_REVISION = "v2"

# The blog's running example. No safety keyword appears in it.
STATE = "When I stop at lights the car pulls hard to the left and the pedal sinks almost to the floor."

DEPARTMENTS = {
    "Service": "Diagnose, repair, or maintain the customer's vehicle.",
    "Parts": "Sell parts or accessories that the customer takes away or installs elsewhere.",
    "Sales": "Buy, lease, or trade in a vehicle.",
    "Warranty Claims": "Refunds, reimbursements, or disputes about warranty coverage. No repair visit.",
}

LEVELS = ["Routine", "Within a week", "Immediately"]
SAFETY = "Is it unsafe to keep driving this vehicle?"

# Two question sets. "bare" lists the departments as plain labels, which is the form the
# walkthrough in the blog traces. "described" passes the criteria descriptions that
# steering_demo.py sends, which makes the prompt longer and the answer different.
SETS = {
    "bare": {
        "department": {
            "type": "choice",
            "instructions": "Which dealership department should handle this?",
            "criteria": {name: None for name in DEPARTMENTS},
        },
        "urgency": {"type": "score", "instructions": "How soon does this need attention?", "criteria": LEVELS},
        "safety_risk": {"type": "noul", "instructions": SAFETY},
    },
    "described": {
        "department": {
            "type": "choice",
            "instructions": "Which dealership department should handle this?",
            "criteria": DEPARTMENTS,
        },
        "urgency": {
            "type": "score",
            "instructions": "How soon does the vehicle need attention?",
            "criteria": LEVELS,
        },
        "safety_risk": {"type": "noul", "instructions": SAFETY},
    },
}


def load():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    sys.modules.setdefault("fla", None)  # its Triton kernel needs a GPU, so use the torch path
    from decider.infer import Decider
    from huggingface_hub import snapshot_download

    return Decider(snapshot_download(DECIDER_MODEL, revision=DECIDER_REVISION))


def slot_hidden(d, ids, slot):
    """Run the backbone on one row and return the hidden state at the answer slot."""
    input_ids = torch.tensor([ids])
    attn = torch.ones_like(input_ids)
    h = d.m.lm.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
    return h[0, slot], tuple(h.shape)


def embed_weight(d):
    """The input embedding matrix, wherever this checkpoint keeps it."""
    m = d.m.lm.model
    return getattr(m, "embed_tokens", None) or m.language_model.embed_tokens


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "bare"
    questions = SETS[which]
    d = load()
    tok = d.m.tok
    print(f"0  question set     {which}")
    rqs, index, items = d._system_one_items(STATE, questions)
    rows = {k: (kind, start, n) for k, kind, start, n in index}
    T = d.T

    choice = items[rows["department"][1]]
    ids, slot = choice["ids"], choice["slots"][0]

    print("1  prompt")
    print(f"   rows built        {len(items)}  (1 choice, 3 isolated levels, 1 noul)")
    print(f"   choice row        {len(ids)} tokens, answer slot {slot}")
    print(f"   last four ids     {ids[-4:]}")
    print(f"   decoded           {[tok.decode([i]) for i in ids[-4:]]}")
    print(f"   nopts             {choice['nopts'][0]}")

    head = d.m.lm.lm_head.weight
    emb = embed_weight(d).weight
    print("\n2  head")
    print(f"   lm_head.weight    {tuple(head.shape)}  {head.shape[0] * head.shape[1]:,} weights")
    print(f"   tied to embed     {head.data_ptr() == emb.data_ptr()}")
    print(f"   letters           {len(d.m.letters)} ids, first six {d.m.letters[:6].tolist()}")
    print(f"   decoded           {[tok.decode([i]) for i in d.m.letters[:6].tolist()]}")
    sliced = head.shape[1] * len(d.m.letters)
    print(f"   slice             ({len(d.m.letters)}, {head.shape[1]}) = {sliced:,} weights")
    print(f"   fraction of head  {100 * sliced / (head.shape[0] * head.shape[1]):.3f}%")

    hs, shape = slot_hidden(d, ids, slot)
    W = head[d.m.letters]
    raw = F.linear(hs, W).float()
    nopts = choice["nopts"][0]
    masked = raw.clone()
    masked[nopts:] = float("-inf")
    names = rqs["department"]["names"]
    p1 = F.softmax(masked, dim=-1)
    pT = F.softmax(masked / T, dim=-1)

    print("\n3  choice")
    print(f"   last_hidden_state {shape}  {d.m.lm.dtype}")
    print(f"   slot hidden state {tuple(hs.shape)}")
    print(f"   raw logits        {[round(x, 2) for x in raw[:6].tolist()]} ...")
    print(f"   masked            {[round(x, 2) for x in masked[:6].tolist()]} ...")
    print(f"   finite entries    {int(torch.isfinite(masked).sum())} of {len(masked)}")
    print(f"   T from config     {T}")
    for i, name in enumerate(names):
        print(f"   {name:<16}  T=1 {p1[i]:.4f}   T={T} {pT[i]:.4f}")

    out = d.system_one(STATE, questions)
    api = out["answers"]
    print("\n4  api")
    print(f"   system_one        {api['department']['probabilities']}")
    print(f"   choice            {api['department']['choice']}")
    print(f"   hand trace        {({n: round(float(pT[i]), 4) for i, n in enumerate(names)})}")
    match = all(
        abs(api["department"]["probabilities"][n] - float(pT[i])) < 5e-4 for i, n in enumerate(names)
    )
    print(f"   agree key for key {match}")

    full = F.linear(hs, head).float()
    pfull = F.softmax(full, dim=-1)
    letter_mass = sum(float(pfull[d.m.letters[i]]) for i in range(nopts))
    top = torch.topk(pfull, 12)
    print("\n5  vocabulary   (unmasked, T=1)")
    print(f"   vocabulary        {head.shape[0]:,} rows")
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"   {tok.decode([idx])!r:<12} {idx:>7}  {prob:.4f}")
    print(f"   four letters      {letter_mass:.4f}")
    print(f"   off the options   {100 * (1 - letter_mass):.2f}%")

    from decider.model import collate

    padded = collate([choice], tok.pad_token_id)
    hp = d.m.lm.model(
        input_ids=padded["input_ids"], attention_mask=padded["attention_mask"]
    ).last_hidden_state[0, slot]
    raw_padded = F.linear(hp, W).float()
    p_padded = F.softmax(raw_padded[:nopts] / T, dim=-1)
    print("\n6  padding      (this is the path system_one runs)")
    print(f"   unpadded length   {len(ids)}")
    print(f"   collate pads to   {padded['input_ids'].shape[1]}")
    print(f"   padded logits     {[round(x, 2) for x in raw_padded[:6].tolist()]} ...")
    print(f"   unpadded answer   {float(pT[0]):.4f}")
    print(f"   padded answer     {float(p_padded[0]):.4f}")
    print(f"   identical         {float(pT[0]) == float(p_padded[0])}")

    pfp = F.softmax(F.linear(hp, head).float(), dim=-1)
    mass_padded = sum(float(pfp[d.m.letters[i]]) for i in range(nopts))
    option_ids = {int(d.m.letters[i]) for i in range(nopts)}
    # Tokens that a reader would call "A" but that are not the option row the head reads.
    top = torch.topk(pfp, 24)
    near = [
        (idx, prob)
        for prob, idx in zip(top.values.tolist(), top.indices.tolist())
        if idx not in option_ids and tok.decode([idx]).strip().strip(".").lower() == "a"
    ]
    print("   full vocabulary, padded, T=1")
    for i in (0, 3):
        idx = int(d.m.letters[i])
        print(f"     {tok.decode([idx])!r:<8} {idx:>7}  {float(pfp[idx]):.4f}")
    for idx, prob in near:
        print(f"     {tok.decode([idx])!r:<8} {idx:>7}  {prob:.4f}")
    print(f"     near misses      {sum(p for _, p in near):.4f}")
    print(f"     four letters     {mass_padded:.4f}")
    print(f"     off the options  {100 * (1 - mass_padded):.2f}%")

    noul = items[rows["safety_risk"][1]]
    nids, nslot = noul["ids"], noul["slots"][0]
    nhs, _ = slot_hidden(d, nids, nslot)
    nraw = F.linear(nhs, head[d.m.letters]).float()
    npT = F.softmax(nraw[:2] / T, dim=-1)
    npad = collate([noul], tok.pad_token_id)
    nhp = d.m.lm.model(
        input_ids=npad["input_ids"], attention_mask=npad["attention_mask"]
    ).last_hidden_state[0, nslot]
    nraw_padded = F.linear(nhp, W).float()
    npT_padded = F.softmax(nraw_padded[:2] / T, dim=-1)
    print("\n7  noul")
    print(f"   row               {len(nids)} tokens, answer slot {nslot}, nopts {noul['nopts'][0]}")
    print(f"   options           {rqs['safety_risk']['options']}")
    print(f"   logits            no {nraw[0]:.2f}   yes {nraw[1]:.2f}")
    print(f"   p                 no {npT[0]:.4f}   yes {npT[1]:.4f}")
    print(f"   padded logits     no {nraw_padded[0]:.2f}   yes {nraw_padded[1]:.2f}")
    print(f"   padded p          no {npT_padded[0]:.4f}   yes {npT_padded[1]:.4f}")
    print(f"   api               {api['safety_risk']}")

    print("\n8  score")
    print(f"   isolated_levels   {d.isolated_levels}")
    print(f"   level rows        {rows['urgency'][2]}")
    for j in range(rows["urgency"][2]):
        item = items[rows["urgency"][1] + j]
        print(f"   level {j} prompt    {len(item['ids'])} tokens")
    print(f"   api               {api['urgency']}")
    listwise = d.system_one(STATE, questions, isolated=False)["answers"]["urgency"]
    print(f"   isolated=False    {listwise}")


if __name__ == "__main__":
    main()
