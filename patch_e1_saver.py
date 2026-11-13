fp = "/home/matteo/hermes-27/scripts/e1_runner.py"
s = open(fp).read()
anchor = 'print("top5 ids:",top5.indices.tolist(),"vals:",[round(v,2) for v in top5.values.tolist()],flush=True)'
assert anchor in s, "anchor not found"
add = anchor + '''
if os.environ.get("E1_SAVE_LOGITS"):
    torch.save({"ids":ids,"logits":logits.half().cpu(),"top5":top5.indices.tolist()},"/tmp/e1_logits.pt")
    print("saved /tmp/e1_logits.pt",flush=True)'''
s = s.replace(anchor, add)
open(fp, "w").write(s)
print("E1_SAVER_PATCH_OK")
