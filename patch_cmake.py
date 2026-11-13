fp = "/home/matteo/hermes-27/CMakeLists.txt"
s = open(fp).read()
anchor = """add_executable(test_mtp_accept architecture/tests/test_mtp_accept.cpp)
target_link_libraries(test_mtp_accept PRIVATE hermes_arch hermes_gguf)
add_test(NAME mtp_accept COMMAND test_mtp_accept)"""
assert anchor in s, "anchor not found"
s = s.replace(anchor, anchor + """
add_executable(test_logits_dump architecture/tests/test_logits_dump.cpp)
target_link_libraries(test_logits_dump PRIVATE hermes_arch hermes_gguf)
# NOT in ctest: heavy full-prefill saver, run only explicitly in queue.""")
open(fp, "w").write(s)
print("CMAKE_PATCH_OK")
