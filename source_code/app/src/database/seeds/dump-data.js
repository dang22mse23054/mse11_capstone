exports.seed = function (knex, Promise) {
	// Deletes ALL existing entries
	return knex('role').del().then(function () {
		// Inserts seed entries
		return knex('role').insert([
			{ 'id': 1, 'name': 'Default' },
			{ 'id': 2, 'name': 'Leader' },
			{ 'id': 3, 'name': 'Manager' },
			{ 'id': 4, 'name': 'Admin' },
			{ 'id': 5, 'name': 'SuperAdmin' },
		]);
	}).then(() => {
		return knex('client_company').del().then(function () {
			// Inserts seed entries
			return knex('client_company').insert([
				{ 'id': 1, 'name': 'Company 1' },
				{ 'id': 2, 'name': 'Company 2' },
				{ 'id': 3, 'name': 'Company 3' },
				{ 'id': 4, 'name': 'Company 4' },
			]);
		});
	}).then(() => {
		return knex('client').del().then(function () {
			// Inserts seed entries
			return knex('client').insert([
				{ 'id': 1, 'name': 'Client 1', 'clientCompanyId': 1 },
				{ 'id': 2, 'name': 'Client 2', 'clientCompanyId': 1 },
				{ 'id': 3, 'name': 'Client 3', 'clientCompanyId': 2 },
				{ 'id': 4, 'name': 'Client 4', 'clientCompanyId': 3 },
				{ 'id': 5, 'name': 'Client 5', 'clientCompanyId': 3 },
				{ 'id': 6, 'name': 'Client 6', 'clientCompanyId': 4 },
				{ 'id': 7, 'name': 'Client 7', 'clientCompanyId': 4 },
				{ 'id': 8, 'name': 'Client 8', 'clientCompanyId': 4 },
			]);
		});
	}).then(() => {
		return knex('client_chat_room').del().then(function () {
			// Inserts seed entries
			return knex('client_chat_room').insert([
				{ 'id': 1, 'clientId': 1, 'roomId': 'room1', chatType: 1 },
				{ 'id': 2, 'clientId': 1, 'roomId': 'room2', chatType: 2 },
				{ 'id': 3, 'clientId': 1, 'roomId': 'room3', chatType: 1, appType: 2 },
				{ 'id': 4, 'clientId': 1, 'roomId': 'room4', chatType: 2, appType: 2 },

				{ 'id': 5, 'clientId': 2, 'roomId': 'room5', chatType: 1 },
				{ 'id': 6, 'clientId': 2, 'roomId': 'room6', chatType: 2 },
				{ 'id': 7, 'clientId': 2, 'roomId': 'room7', chatType: 1, appType: 2 },
				{ 'id': 8, 'clientId': 2, 'roomId': 'room8', chatType: 2, appType: 2 },

				{ 'id': 9, 'clientId': 3, 'roomId': 'room9', chatType: 1 },

				{ 'id': 10, 'clientId': 4, 'roomId': 'room10', chatType: 2 },

				{ 'id': 11, 'clientId': 5, 'roomId': 'room11', chatType: 1, appType: 2 },
				{ 'id': 12, 'clientId': 5, 'roomId': 'room12', chatType: 2, appType: 2 },
			]);
		});
	}).then(() => {
		return knex('category').del().then(function () {
			// Inserts seed entries
			return knex('category').insert([
				{ 'id': 1, 'name': 'category_1', parentId: null, isUseMedia: true },
				{ 'id': 2, 'name': 'category_2', parentId: 1, isUseMedia: true },
				{ 'id': 3, 'name': 'category_3', parentId: 1, isUseMedia: true },
				{ 'id': 4, 'name': 'category_4', parentId: null, isUseMedia: true },
				{ 'id': 5, 'name': 'category_5', parentId: 4, isUseMedia: true },
				{ 'id': 6, 'name': 'category_6', parentId: null, isUseMedia: true },
				{ 'id': 7, 'name': 'category_7', parentId: null, isUseMedia: true },
				{ 'id': 8, 'name': 'category_8', parentId: 7, isUseMedia: true },
				{ 'id': 9, 'name': 'category_9', parentId: 8, isUseMedia: true },
			]);
		});
	}).then(() => {
		return knex('media').del().then(function () {
			// Inserts seed entries
			return knex('media').insert([
				{ 'id': 1, 'name': 'media_1' },
				{ 'id': 2, 'name': 'media_2' },
				{ 'id': 3, 'name': 'media_3' },
				{ 'id': 4, 'name': 'media_4' },
				{ 'id': 5, 'name': 'media_5' },
				{ 'id': 6, 'name': 'media_6' },
				{ 'id': 7, 'name': 'media_7' },
				{ 'id': 8, 'name': 'media_8' },
				{ 'id': 9, 'name': 'media_9' },
			]);
		});
	}).then(() => {
		return knex('organization').del().then(function () {
			// Inserts seed entries
			return knex('organization').insert([
				{ 'id': 1, 'mapId': 1, 'name': 'company_1', 'level': 0, parentMapId: null },
				{ 'id': 2, 'mapId': 2, 'name': 'sub(1)_2', 'level': 1, parentMapId: 1 },
				{ 'id': 3, 'mapId': 3, 'name': 'sub(1)_3', 'level': 2, parentMapId: 1 },
				{ 'id': 4, 'mapId': 4, 'name': 'company_4', 'level': 0, parentMapId: null },
				{ 'id': 5, 'mapId': 5, 'name': 'sub(4)_5', 'level': 1, parentMapId: 4 },
				{ 'id': 6, 'mapId': 6, 'name': 'company_6', 'level': 0, parentMapId: null },
				{ 'id': 7, 'mapId': 7, 'name': 'company_7', 'level': 0, parentMapId: null },
				{ 'id': 8, 'mapId': 8, 'name': 'sub(7)_8', 'level': 1, parentMapId: 7 },
				{ 'id': 9, 'mapId': 9, 'name': 'sub(7)_9', 'level': 2, parentMapId: 8 },
			]);
		});
	});
};