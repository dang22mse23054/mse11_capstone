exports.seed = function (knex, Promise) {
	// Deletes ALL existing entries
	return knex('category').del()
		.then(function () {
			// Inserts seed entries
			return knex('category').insert([
				// CHILDREN: 	(00~12) => 0
				{ id: 1, name: 'Male_Children', gender: 0, age: 0 },
				{ id: 2, name: 'Female_Children', gender: 1, age: 0 },
				
				// TEENAGERS: 	(13~17) => 1
				{ id: 3, name: 'Male_Teenager', gender: 0, age: 1 },
				{ id: 4, name: 'Female_Teenager', gender: 1, age: 1 },
				
				// ADULT: 		(18~44) => 2
				{ id: 5, name: 'Male_Adule', gender: 0, age: 2 },
				{ id: 6, name: 'Female_Adule', gender: 1, age: 2 },
				
				// MIDDLE_AGED:(45~60) => 3
				{ id: 7, name: 'Male_MiddleAge', gender: 0, age: 3 },
				{ id: 8, name: 'Female_MiddleAge', gender: 1, age: 3 },
				
				// ELDERLY: 	(61~12) => 4
				{ id: 9, name: 'Male_Elderly', gender: 0, age: 4 },
				{ id: 10, name: 'Female_Elderly', gender: 1, age: 4 },
			]);
		});
};