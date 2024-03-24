import { blue, red, grey } from '@material-ui/core/colors';
import { mdiPlayCircleOutline, mdiStopCircleOutline, mdiPauseCircleOutline } from '@mdi/js';

export const VideoStatus = {
	PAUSED: { iconFixedColor: grey[600], iconName: mdiPauseCircleOutline, label: 'Paused', value: 0 },
	PLAYING: { iconFixedColor: blue[600], iconName: mdiPlayCircleOutline, label: 'Playing', value: 1 },
	STOPPED: { iconFixedColor: red[600], iconName: mdiStopCircleOutline, label: 'Stopped', value: 2 },
} as const;

export const VideoStatusMap = {
	[VideoStatus.STOPPED.value]: VideoStatus.STOPPED,
	[VideoStatus.PLAYING.value]: VideoStatus.PLAYING,
	[VideoStatus.PAUSED.value]: VideoStatus.PAUSED,
} as const;