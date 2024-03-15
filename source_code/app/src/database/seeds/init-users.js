exports.seed = function (knex, Promise) {
	// Deletes ALL existing entries
	return knex('users').del()
		.then(function () {
			// Inserts seed entries
			return knex('users').insert([
				{ 
					id: 1,
					uid: 'dt0294',
					fullname: 'Ha Hai Dang',
				}
			]);
		});
};
