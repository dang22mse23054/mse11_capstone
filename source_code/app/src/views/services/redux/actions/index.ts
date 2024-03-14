// Dispatch
import { AuthAction } from './Auth';
import { SideBarAction } from './SideBar';

import { ActionTypes } from './action-type';
export * from './action-object';

const Actions = {
	AuthAction,
	SideBarAction,
};

export { Actions, ActionTypes };
