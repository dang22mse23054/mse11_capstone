// https://stackoverflow.com/questions/53312320/webpack-dev-server-ts-loader-not-reloads-when-interface-changed
import {
	IPagingObj,
	ICursorInput,
	IGraphqlPageInfo,
	ICAUserOption,
} from 'interfaceDir';
import { Moment } from 'moment';
import { DropDownOption } from 'compDir/DropDownSelect';
import { ChannelMemberStatusValues } from 'rootDir/constants';

export interface IActionObj {
  type: string;
  obj?: any;
}

export interface IAuthActionObj extends IActionObj {
  authData?: any;
  isFirstLoading?: boolean;
}
