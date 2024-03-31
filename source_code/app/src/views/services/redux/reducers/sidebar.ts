import { ActionTypes } from '../actions';
import { IActionObj } from '../actions/action-object';

export interface IReducer {
	openSideBar: boolean
}

const initialState: IReducer = {
	openSideBar: false
};

const handler = (state = initialState, action: IActionObj) => {
	switch (action.type) {
		//-------- Schedule setting action --------//
		case ActionTypes.SideBar.TOGGLE_SIDE_BAR:
			return toggle(state);
			break;

		default:
			return state;
			break;
	}
};

const toggle = (state) => {
	return {
		...state,
		openSideBar: !state.openSideBar
	};
};

export default handler;