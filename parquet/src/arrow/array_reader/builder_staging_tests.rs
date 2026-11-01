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
        let metadata = FileMetaData::new(1, 0, None, None, descriptor, None);
        Self {
            metadata: ParquetMetaData::new(metadata, vec![]),
            opened: RefCell::new(vec![]),
        }
    }

    fn levels(&self, schema: &Schema, virtuals: &[FieldRef]) -> crate::arrow::schema::FieldLevels {
        parquet_to_arrow_field_levels_with_virtual(
            self.metadata.file_metadata().schema_descr(),
            ProjectionMask::all(),
            Some(schema.fields()),
            virtuals,
        )
        .unwrap()
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

fn field(name: &str, data_type: DataType) -> FieldRef {
    Arc::new(
        Field::new(name, data_type, true)
            .with_metadata(HashMap::from([("tag".into(), name.into())])),
    )
}

fn fixture_schema() -> Schema {
    let item = field("item", DataType::Utf8);
    let entries = Arc::new(Field::new(
        "entries",
        DataType::Struct(
            vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Binary, true),
            ]
            .into(),
        ),
        false,
    ));
    Schema::new_with_metadata(
        vec![
            field("text", DataType::Utf8),
            field("binary", DataType::Binary),
            field("large", DataType::LargeUtf8),
            field("large_binary", DataType::LargeBinary),
            field("view", DataType::Utf8View),
            field("binary_view", DataType::BinaryView),
            field(
                "dictionary",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            ),
            field("structure", DataType::Struct(vec![item.clone()].into())),
            field("list", DataType::List(item.clone())),
            field("large_list", DataType::LargeList(item.clone())),
            field("fixed_list", DataType::FixedSizeList(item.clone(), 2)),
            field("list_view", DataType::ListView(item.clone())),
            field("large_list_view", DataType::LargeListView(item)),
            field("map", DataType::Map(entries, true)),
        ],
        HashMap::from([("schema-tag".into(), "omitted-by-reader-contract".into())]),
    )
}

#[test]
fn test_flat_staging_preserves_nested_types_and_cache_schemas() {
    let schema = fixture_schema();
    let groups = CountingRowGroups::new(&schema);
    let levels = groups.levels(&schema, &[]);
    let metrics = ArrowReaderMetrics::disabled();
    let mask = ProjectionMask::all();
    let cache = Arc::new(RwLock::new(RowGroupCache::new(8, usize::MAX)));
    let mut cache_types = vec![];

    for role in [CacheRole::Producer, CacheRole::Consumer] {
        groups.opened.borrow_mut().clear();
        let options = CacheOptions {
            projection_mask: &mask,
            cache: &cache,
            role,
        };
        let (reader, logical) = ArrayReaderBuilder::new(&groups, &metrics)
            .with_cache_options(Some(&options))
            .with_byte_array_staging(true)
            .build_array_reader_with_logical_schema(levels.levels.as_ref(), &mask)
            .unwrap();
        assert_eq!(logical.fields(), schema.fields());
        assert!(logical.metadata().is_empty());
        let DataType::Struct(decoded) = reader.get_data_type() else {
            unreachable!()
        };
        assert_eq!(
            decoded[0],
            Arc::new(schema.field(0).clone().with_data_type(DataType::LargeUtf8))
        );
        assert_eq!(
            decoded[1],
            Arc::new(
                schema
                    .field(1)
                    .clone()
                    .with_data_type(DataType::LargeBinary)
            )
        );
        assert_eq!(&decoded[2..], &schema.fields()[2..]);
        let columns = groups.metadata.file_metadata().schema_descr().num_columns();
        assert_eq!(*groups.opened.borrow(), (0..columns).collect::<Vec<_>>());
        cache_types.push(reader.get_data_type().clone());
    }
    assert_eq!(cache_types[0], cache_types[1]);

    let (reader, logical) = ArrayReaderBuilder::new(&groups, &metrics)
        .build_array_reader_with_logical_schema(levels.levels.as_ref(), &mask)
        .unwrap();
    assert_eq!(
        reader.get_data_type(),
        &DataType::Struct(logical.fields().clone())
    );
}

#[test]
fn test_flat_staging_projection_uses_positions_with_duplicate_names() {
    let nested = field(
        "nested",
        DataType::Struct(
            vec![
                field("same", DataType::Utf8),
                field("drop", DataType::Int32),
            ]
            .into(),
        ),
    );
    let schema = Schema::new(vec![
        field("same", DataType::Utf8),
        field("same", DataType::LargeUtf8),
        nested.clone(),
    ]);
    let groups = CountingRowGroups::new(&schema);
    let levels = groups.levels(&schema, &[]);
    let metrics = ArrowReaderMetrics::disabled();
    let descriptor = groups.metadata.file_metadata().schema_descr();
    let projected_nested = Arc::new(
        nested
            .as_ref()
            .clone()
            .with_data_type(DataType::Struct(vec![field("same", DataType::Utf8)].into())),
    );

    for flat in [0, 1] {
        let mask = ProjectionMask::leaves(descriptor, [flat, 2]);
        let (reader, logical) = ArrayReaderBuilder::new(&groups, &metrics)
            .with_byte_array_staging(true)
            .build_array_reader_with_logical_schema(levels.levels.as_ref(), &mask)
            .unwrap();
        assert_eq!(
            logical.fields(),
            &Fields::from(vec![
                schema.fields()[flat].clone(),
                projected_nested.clone(),
            ])
        );
        let DataType::Struct(decoded) = reader.get_data_type() else {
            unreachable!()
        };
        assert_eq!(decoded[0].data_type(), &DataType::LargeUtf8);
        assert_eq!(decoded[1], projected_nested);
    }
}

#[test]
fn test_flat_staging_empty_and_virtual_projections() {
    let schema = Schema::new(vec![field("text", DataType::Utf8)]);
    let groups = CountingRowGroups::new(&schema);
    let metrics = ArrowReaderMetrics::disabled();
    let mask = ProjectionMask::leaves(groups.metadata.file_metadata().schema_descr(), []);
    let levels = groups.levels(&schema, &[]);
    for field in [None, levels.levels.as_ref()] {
        let (reader, logical) = ArrayReaderBuilder::new(&groups, &metrics)
            .with_byte_array_staging(true)
            .build_array_reader_with_logical_schema(field, &mask)
            .unwrap();
        assert!(logical.fields().is_empty());
        assert_eq!(reader.get_data_type(), &DataType::Struct(Fields::empty()));
    }
    let row_number =
        Arc::new(Field::new("row_number", DataType::Int64, false).with_extension_type(RowNumber));
    let levels = groups.levels(&schema, std::slice::from_ref(&row_number));
    let (reader, logical) = ArrayReaderBuilder::new(&groups, &metrics)
        .with_byte_array_staging(true)
        .with_parquet_metadata(&groups.metadata)
        .build_array_reader_with_logical_schema(levels.levels.as_ref(), &mask)
        .unwrap();
    assert_eq!(logical.fields(), &Fields::from(vec![row_number]));
    assert_eq!(
        reader.get_data_type(),
        &DataType::Struct(logical.fields().clone())
    );
    assert!(groups.opened.borrow().is_empty());
}
