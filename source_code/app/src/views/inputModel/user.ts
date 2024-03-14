import { IUserInput } from 'rootDir/interfaces/user';

export class UserInput implements IUserInput {
	id: number;
	fullname?: string | undefined;
	chatworkAccId?: string | undefined;
	chatworkAccName?: string | undefined;
	slackAccId?: string | undefined;

	constructor(option: IUserInput = {}) {
		this.id = option.id ? Number(option.id) : undefined;
		this.fullname = option.fullname;
		this.chatworkAccId = option.chatworkAccId;
		this.chatworkAccName = option.chatworkAccName;
		this.slackAccId = option.slackAccId;
	}
}
