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

use super::*;
use crate::arrow::array_reader::test_util::EmptyPageIterator;
use crate::arrow::schema::{
    ArrowSchemaConverter, parquet_to_arrow_field_levels_with_virtual, virtual_type::RowNumber,
};
use crate::column::page::PageIterator;
use crate::file::metadata::{FileMetaData, RowGroupMetaData};
use arrow_schema::{Field, FieldRef};
use std::cell::RefCell;
use std::collections::HashMap;

struct CountingRowGroups {
    metadata: ParquetMetaData,
    opened: RefCell<Vec<usize>>,
}

impl CountingRowGroups {
    fn new(schema: &Schema) -> Self {
        let descriptor = Arc::new(ArrowSchemaConverter::new().convert(schema).unwrap());
        let file_metadata = FileMetaData::new(1, 0, None, None, descriptor, None);
        Self {
            metadata: ParquetMetaData::new(file_metadata, vec![]),
            opened: RefCell::new(vec![]),
        }
    }

    fn mask(&self, names: &[&str]) -> ProjectionMask {
        let descriptor = self.metadata.file_metadata().schema_descr();
        ProjectionMask::leaves(
            descriptor,
            descriptor
                .columns()
                .iter()
                .enumerate()
                .filter_map(|(index, column)| names.contains(&column.name()).then_some(index)),
        )
    }
}

impl RowGroups for CountingRowGroups {
    fn num_rows(&self) -> usize {
        0
    }

    fn column_chunks(&self, column: usize) -> Result<Box<dyn PageIterator>> {
        self.opened.borrow_mut().push(column);
        Ok(Box::new(EmptyPageIterator::default()))
    }

    fn row_groups(&self) -> Box<dyn Iterator<Item = &RowGroupMetaData> + '_> {
        Box::new(self.metadata.row_groups().iter())
    }

    fn metadata(&self) -> &ParquetMetaData {
        &self.metadata
    }
}

fn field(name: &str, data_type: DataType, nullable: bool) -> FieldRef {
    Arc::new(
        Field::new(name, data_type, nullable)
            .with_metadata(HashMap::from([("owner".into(), name.into())])),
    )
}

fn fixture_schema() -> Schema {
    let nested = field(
        "item",
        DataType::Struct(
            vec![
                field("x", DataType::Utf8, true),
                field("y", DataType::Int32, false),
            ]
            .into(),
        ),
        true,
    );
    let entries = field(
        "entries",
        DataType::Struct(
            vec![
                field("key", DataType::Utf8, false),
                field(
                    "value",
                    DataType::Struct(
                        vec![
                            field("keep", DataType::Utf8, true),
                            field("drop", DataType::Int32, true),
                        ]
                        .into(),
                    ),
                    true,
                ),
            ]
            .into(),
        ),
        false,
    );
    Schema::new_with_metadata(
        vec![
            field("text", DataType::Utf8, true),
            field("binary", DataType::Binary, true),
            field("requested_view", DataType::Utf8View, false),
            field("requested_binary_view", DataType::BinaryView, true),
            field("large", DataType::LargeUtf8, true),
            field(
                "dictionary",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
            field("decimal", DataType::Decimal128(9, 2), true),
            field("nested", DataType::List(nested), true),
            field("map", DataType::Map(entries, true), true),
        ],
        HashMap::from([("schema-owner".into(), "omitted-by-reader-contract".into())]),
    )
}

fn original_levels(
    groups: &CountingRowGroups,
    schema: &Schema,
    virtuals: &[FieldRef],
) -> crate::arrow::schema::FieldLevels {
    parquet_to_arrow_field_levels_with_virtual(
        groups.metadata.file_metadata().schema_descr(),
        ProjectionMask::all(),
        Some(schema.fields()),
        virtuals,
    )
    .unwrap()
}

#[test]
fn test_byte_array_staging_builder_preserves_original_schema_and_builds_once() {
    let schema = fixture_schema();
    let groups = CountingRowGroups::new(&schema);
    let levels = original_levels(&groups, &schema, &[]);
    let metrics = ArrowReaderMetrics::disabled();
    let (reader, logical) = ArrayReaderBuilder::new(&groups, &metrics)
        .with_byte_array_staging(true)
        .build_array_reader_with_schema(levels.levels.as_ref(), &ProjectionMask::all())
        .unwrap();
    assert_eq!(logical.fields(), schema.fields());
    assert!(logical.metadata().is_empty());
    let DataType::Struct(staging) = reader.get_data_type() else {
        panic!("expected Struct")
    };
    assert_eq!(staging[0].data_type(), &DataType::LargeUtf8);
    assert_eq!(staging[1].data_type(), &DataType::LargeBinary);
    for index in 2..7 {
        assert_eq!(
            staging[index],
            schema.fields()[index],
            "original non-staged field {index}"
        );
    }
    let DataType::List(item) = staging[7].data_type() else {
        panic!("expected List")
    };
    let DataType::Struct(children) = item.data_type() else {
        panic!("expected item Struct")
    };
    assert_eq!(children[0].data_type(), &DataType::LargeUtf8);
    assert_eq!(children[1].data_type(), &DataType::Int32);
    assert_eq!(
        groups.opened.borrow().len(),
        groups.metadata.file_metadata().schema_descr().num_columns()
    );
    let mut opened = groups.opened.borrow().clone();
    opened.sort_unstable();
    opened.dedup();
    assert_eq!(
        opened.len(),
        groups.opened.borrow().len(),
        "reader construction must not run twice"
    );

    groups.opened.borrow_mut().clear();
    let (reader, logical) = ArrayReaderBuilder::new(&groups, &metrics)
        .build_array_reader_with_schema(levels.levels.as_ref(), &ProjectionMask::all())
        .unwrap();
    assert_eq!(
        reader.get_data_type(),
        &DataType::Struct(logical.fields().clone())
    );
}

#[test]
fn test_byte_array_staging_builder_projects_original_nested_fields_and_map() {
    let schema = fixture_schema();
    let groups = CountingRowGroups::new(&schema);
    let levels = original_levels(&groups, &schema, &[]);
    let metrics = ArrowReaderMetrics::disabled();
    let mask = groups.mask(&["x", "key", "keep"]);
    let (reader, logical) = ArrayReaderBuilder::new(&groups, &metrics)
        .with_byte_array_staging(true)
        .build_array_reader_with_schema(levels.levels.as_ref(), &mask)
        .unwrap();
    let expected_nested = field(
        "nested",
        DataType::List(field(
            "item",
            DataType::Struct(vec![field("x", DataType::Utf8, true)].into()),
            true,
        )),
        true,
    );
    let expected_map = field(
        "map",
        DataType::Map(
            field(
                "entries",
                DataType::Struct(
                    vec![
                        field("key", DataType::Utf8, false),
                        field(
                            "value",
                            DataType::Struct(vec![field("keep", DataType::Utf8, true)].into()),
                            true,
                        ),
                    ]
                    .into(),
                ),
                false,
            ),
            true,
        ),
        true,
    );
    assert_eq!(
        logical.fields(),
        &Fields::from(vec![expected_nested, expected_map])
    );
    assert_eq!(groups.opened.borrow().len(), 3);
    let DataType::Struct(staging) = reader.get_data_type() else {
        unreachable!()
    };
    let DataType::Map(entries, true) = staging[1].data_type() else {
        panic!("expected sorted Map")
    };
    let DataType::Struct(entries) = entries.data_type() else {
        unreachable!()
    };
    assert_eq!(entries[0].data_type(), &DataType::LargeUtf8);

    groups.opened.borrow_mut().clear();
    assert!(
        ArrayReaderBuilder::new(&groups, &metrics)
            .with_byte_array_staging(true)
            .build_array_reader_with_schema(levels.levels.as_ref(), &groups.mask(&["keep"]))
            .is_err()
    );
    assert!(groups.opened.borrow().is_empty());
}

#[test]
fn test_byte_array_staging_builder_empty_and_virtual_projections() {
    let schema = fixture_schema();
    let groups = CountingRowGroups::new(&schema);
    let metrics = ArrowReaderMetrics::disabled();
    let mask = groups.mask(&[]);
    let levels = original_levels(&groups, &schema, &[]);
    let (reader, logical) = ArrayReaderBuilder::new(&groups, &metrics)
        .with_byte_array_staging(true)
        .build_array_reader_with_schema(levels.levels.as_ref(), &mask)
        .unwrap();
    assert!(logical.fields().is_empty());
    assert_eq!(reader.get_data_type(), &DataType::Struct(Fields::empty()));
    assert!(groups.opened.borrow().is_empty());

    let row_number =
        Arc::new(Field::new("row_number", DataType::Int64, false).with_extension_type(RowNumber));
    let levels = original_levels(&groups, &schema, std::slice::from_ref(&row_number));
    let (reader, logical) = ArrayReaderBuilder::new(&groups, &metrics)
        .with_byte_array_staging(true)
        .with_parquet_metadata(&groups.metadata)
        .build_array_reader_with_schema(levels.levels.as_ref(), &mask)
        .unwrap();
    assert_eq!(logical.fields(), &Fields::from(vec![row_number]));
    assert_eq!(
        reader.get_data_type(),
        &DataType::Struct(logical.fields().clone())
    );
    assert!(groups.opened.borrow().is_empty());
}

#[test]
fn test_byte_array_staging_builder_keeps_list_view_subtrees_native() {
    let nested_item = field(
        "item",
        DataType::Struct(
            vec![
                field("deep_text", DataType::Utf8, true),
                field("deep_binary", DataType::Binary, true),
            ]
            .into(),
        ),
        true,
    );
    let schema = Schema::new(vec![
        field("list_view", DataType::ListView(nested_item.clone()), true),
        field(
            "large_list_view",
            DataType::LargeListView(nested_item),
            true,
        ),
        field("outside", DataType::Utf8, true),
    ]);
    let groups = CountingRowGroups::new(&schema);
    let levels = original_levels(&groups, &schema, &[]);
    let metrics = ArrowReaderMetrics::disabled();
    let (reader, logical) = ArrayReaderBuilder::new(&groups, &metrics)
        .with_byte_array_staging(true)
        .build_array_reader_with_schema(levels.levels.as_ref(), &ProjectionMask::all())
        .unwrap();
    assert_eq!(logical.fields(), schema.fields());
    let DataType::Struct(staging) = reader.get_data_type() else {
        unreachable!()
    };
    assert_eq!(staging[0], schema.fields()[0]);
    assert_eq!(staging[1], schema.fields()[1]);
    assert_eq!(
        staging[2].data_type(),
        &DataType::LargeUtf8,
        "only ListView subtrees are left unstaged"
    );
}
