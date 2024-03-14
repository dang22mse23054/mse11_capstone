exports.up = function (knex, Promise) {
	return knex.schema
		.dropTableIfExists('video')
		.createTable('video', function (table) {
			table.increments('id').primary();
			table.string('title', 200).notNullable();
			table.string('path', 500).notNullable();

			table.timestamp('deletedAt').nullable();
			// table.timestamps(); // this option cannot set default value to createdAt & updatedAt columns
			table.timestamp('createdAt').notNullable().defaultTo(knex.fn.now());
			table.timestamp('updatedAt').notNullable().defaultTo(knex.fn.now());
		});
};

exports.down = function (knex, Promise) {
	return knex.schema.dropTable('video');
};
