exports.up = function (knex, Promise) {
	return knex.schema
		.dropTableIfExists('category')
		.createTable('category', function (table) {
			table.increments('id').primary();
			table.string('name', 200).notNullable();

			table.integer('gender').unsigned().nullable()
				.comment('Male => 0, Female => 1');;
			table.integer('age').unsigned().nullable()
				.comment('CHILDREN (00~12) => 0, TEENAGERS (13~17) => 1, ADULT (18~44) => 2, MIDDLE_AGED (45~60) => 3, ELDERLY (61~12) => 4');;
				

			table.timestamp('deletedAt').nullable();
			// table.timestamps(); // this option cannot set default value to createdAt & updatedAt columns
			table.timestamp('createdAt').notNullable().defaultTo(knex.fn.now());
			table.timestamp('updatedAt').notNullable().defaultTo(knex.fn.now());
		});
};

exports.down = function (knex, Promise) {
	return knex.schema.dropTable('category');
};
