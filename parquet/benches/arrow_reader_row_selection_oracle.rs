// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Forced-policy oracle for Parquet `RowSelection` execution.
//!
//! The default protocol is the complete, adaptive matrix. Use `--quick` for
//! the curated C0/C3 smoke subset intended for inexpensive validation.

mod row_selection_policy_common;

fn main() {
    row_selection_policy_common::oracle::main();
}
