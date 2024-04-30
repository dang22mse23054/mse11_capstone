const Type = {
	genderDetail: (obj, args, context, info) => {
		try {
			if (obj.gender) {
				parsedObj = JSON.parse(obj.gender)
				return {
					male: parsedObj[0],
					female: parsedObj[1]
				};
			}
		} catch (error) {
			console.error(error)
		}
		
		return null;
	},

	ageDetail: (obj, args, context, info) => {
		try {
			if (obj.age) {
				parsedObj = JSON.parse(obj.age)
				return {
					children: parsedObj[0],
					teenager: parsedObj[1],
					adult: parsedObj[2],
					middleAge: parsedObj[3],
					elderly: parsedObj[4],
				};
			}

		} catch (error) {
			console.error(error)
		}
		return null;
	},

	happyDetail: (obj, args, context, info) => {
		try {
			if (obj.happy) {
				parsedObj = JSON.parse(obj.happy)
				genderObj = parsedObj[0]
				ageObj = parsedObj[1]
				return {
					gender: {
						male: genderObj[0],
						female: genderObj[1]
					},
					age: {
						children: ageObj[0],
						teenager: ageObj[1],
						adult: ageObj[2],
						middleAge: ageObj[3],
						elderly: ageObj[4],
					}
				};
			}

		} catch (error) {
			console.error(error)
		}
		return null;
	},
};

module.exports = { Type };