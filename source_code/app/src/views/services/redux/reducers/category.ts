import { ActionTypes } from '../actions';
import { ICategoryDefaultAO } from '../actions/action-object';

export interface IReducer extends ICategoryDefaultAO {
	setting: ICategorySettingAO
}

const initialState: IReducer = {
	setting: {
		categoryList: [],
	},
	initCategoryList: []
};

const handler = (state = initialState, action: IActionObj) => {
	switch (action.type) {
		//-------- Category view action --------//
		case ActionTypes.Category.INIT_CATEGORY_LIST:
			return setInitCategoryList(state, (action as ICategoryDefaultAO).initCategoryList);
			break;

		default:
			return state;
			break;
	}
};

const setInitCategoryList = (state: IReducer, initCategoryList): IReducer => {
	return {
		...state,
		initCategoryList
	};
};

export default handler;