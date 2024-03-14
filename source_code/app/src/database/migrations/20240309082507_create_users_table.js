exports.up = function (knex, Promise) {
	return knex.schema
		.dropTableIfExists('users')
		.createTable('users', function (table) {
			table.increments('id').primary();
			table.string('uid', 20).unique().notNullable();

			// CHILDREN: 	(00~12) => 0
			// TEENAGERS: 	(13~17) => 1
			// ADULT: 		(18~44) => 2
			// MIDDLE_AGED:(45~60) => 3
			// ELDERLY: 	(61~12) => 4
			table.string('fullname', 50).notNullable();
			
			table.timestamp('lastAccess').nullable();
			table.timestamp('deletedAt').nullable();
			// table.timestamps(); // this option cannot set default value to createdAt & updatedAt columns
			table.timestamp('createdAt').notNullable().defaultTo(knex.fn.now());
			table.timestamp('updatedAt').notNullable().defaultTo(knex.fn.now());

			table.index(['uid', 'lastAccess', 'deletedAt'], 'users_table_index');
		});
};

exports.down = function (knex, Promise) {
	return knex.schema.dropTable('users');
};
