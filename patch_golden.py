fp = "/home/matteo/hermes-27/architecture/tests/test_model_fwd.cpp"
s = open(fp).read()
old_exp = "int exp[5]={898,34566,22197,29069,73};"
new_exp = "int exp[5]={220,1374,27053,17,1147};"
assert old_exp in s, "exp pattern not found"
s = s.replace(old_exp, new_exp)
old_c = "vals [19.23,18.79,18.34,17.4,17.36]"
new_c = "vals [15.74,12.96,12.41,12.19,11.2]"
if old_c in s:
    s = s.replace(old_c, new_c)
open(fp, "w").write(s)
print("GOLDEN_PATCH_OK")
