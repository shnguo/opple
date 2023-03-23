CREATE table if NOT EXISTS opl_forcasting_month(
    id String,
    unique_id String,
    year UInt16,
    month UInt8,
    year_month Date,
    y Float32,
    price Float32,
    mount Float32,
    type String,
    model String,
    timestamp DateTime,
)
ENGINE = ReplacingMergeTree()
order by (id,unique_id,year,month,model);

CREATE table if not exists opl_ori_data(
	id String,
	unique_id String,
	description String,
	y Float32,
	price Float32,
	mount Float32,
	year_month String,
	date String,
	timestamp DateTime,
)ENGINE = ReplacingMergeTree()
order by (id,unique_id,date);

CREATE table if NOT EXISTS opl_pre_data_month(
    unique_id String,
    year UInt16,
    month UInt8,
    year_month Date,
    y Float32,
    price Float32,
    mount Float32,
    timestamp DateTime,
)
ENGINE = ReplacingMergeTree()
order by (unique_id,year,month);
