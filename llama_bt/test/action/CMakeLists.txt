# Test for the generate response action
find_package(ament_index_cpp REQUIRED)
ament_add_gtest(test_generate_response_action
  test_generate_response_action.cpp
)
target_link_libraries(test_generate_response_action
  llama_generate_response_action_bt_node
)
ament_target_dependencies(test_generate_response_action
  ${dependencies}
  ament_index_cpp
)