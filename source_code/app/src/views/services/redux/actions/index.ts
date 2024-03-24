// Dispatch
import { AuthAction } from './Auth';
import { SideBarAction } from './SideBar';
import { VideoAction } from './Video';
import { CategoryAction } from './Category';

import { ActionTypes } from './action-type';
export * from './action-object';

const Actions = {
	AuthAction,
	SideBarAction,
	VideoAction,
	CategoryAction,
};

export { Actions, ActionTypes };
