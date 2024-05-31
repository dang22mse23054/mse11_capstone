import { ICategoryAO } from '../action-object';
import ActionTypes from './types';

export class CategoryAction {

	public static initCategory(initCategoryList): ICategoryAO {
		return {
			type: ActionTypes.INIT_CATEGORY_LIST,
			initCategoryList
		};
	}
}