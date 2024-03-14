import { IActionObj } from '../action-object';
import ActionTypes from './types';

export class SideBarAction {

	public static toggleSideBar(): IActionObj {
		return {
			type: ActionTypes.TOGGLE_SIDE_BAR,
		};
	}
}